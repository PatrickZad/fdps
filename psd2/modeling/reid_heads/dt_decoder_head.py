from psd2.modeling.transformer.deformable import (
    DeformableTransformerDecoderLayer,
    _get_clones,
    _get_activation_fn,
)
import os
from .base_reid_head import ReidHeadBase
import torch
from psd2.structures.boxes import box_xyxy_to_cxcywh
import torch.nn.functional as tF
import random
from psd2.layers.ops.modules import MSDeformAttn
from psd2.layers.ops.functions import MSDeformAttnFunction
from torch import nn
from psd2.utils.events import get_event_storage
from psd2.utils.visualizer import Visualizer
import numpy as np
import colorsys
import torchvision.transforms.functional as tvF
from torch.nn.init import xavier_uniform_, constant_, uniform_
from psd2.utils import comm

sa_modes = ["original", "res_linear", "linear", "res_dynamic", "dynamic"]
sa_prompt = ["img-wise", "inst-wise"]
ca_mode = ["none", "mask", "box", "box_ref"]
prompt_out_feat = ["instance", "prompt", "inst+prompt"]


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class CompBN1D(nn.BatchNorm1d):
    def forward(self, x):
        """
        x: batch x seq x channel
        """
        if x.dim() == 3:
            x_t = x.transpose(1, 2)
            x_n = super().forward(x_t)
            return x_n.transpose(1, 2)
        else:
            return super().forward(x)


class BoxMSDeformAttn(MSDeformAttn):
    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        r_samp_pt=False,
        box_ref=False,
    ):
        super().__init__(d_model, n_levels, n_heads, n_points, r_samp_pt)
        if box_ref:
            # multi scale box refiner TODO output for multi-layer
            self.linear_box_weight = nn.Parameter(
                torch.zeros(self.n_levels * 4, d_model)
            )
            self.linear_box_bias = nn.Parameter(torch.zeros(self.n_levels * 4))
            nn.init.constant_(self.linear_box_weight, 0.0)
            nn.init.constant_(self.linear_box_bias, 0.0)
            self.box_refiner = self._ms_box_refiner
        else:
            self.box_refiner = self._duplicate_box_refiner

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        # init with n_heads x n_points grid
        grid_x = torch.arange(1, self.n_points + 1).view(1, -1).repeat(
            self.n_heads, 1
        ) / (
            self.n_points + 1
        )  #  n_heads x n_points
        grid_y = torch.arange(1, self.n_heads + 1).view(-1, 1).repeat(
            1, self.n_points
        ) / (
            self.n_heads + 1
        )  #  n_heads x n_points
        grid_init = torch.stack([grid_x, grid_y], dim=-1)  # n_heads x n_points x 2
        grid_init_invsg = _inverse_sigmoid(grid_init)
        grid_init_invsg = grid_init_invsg.unsqueeze(1).repeat(
            1, self.n_levels, 1, 1
        )  # n_heads x n_levels x n_points x 2
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init_invsg.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _duplicate_box_refiner(self, query, ref_box, box_normalizer):
        """
        ref_boxes: N x Length_{query} x n_scale x 4, xyxy box in padding rel
        """
        nb, nseq = ref_box.shape[:2]
        return ref_box.reshape(nb, nseq, 1, -1, 1, 4).repeat(
            1, 1, self.n_heads, 1, self.n_points, 1
        )

    def _ms_box_refiner(self, query, ref_box, box_normalizer):
        """
        ref_boxes: N x Length_{query} x n_levels x 4, xyxy box in padding rel
        box_normalizer: n_levels x 2 (n_levels, 2), [(W_0, H_0), (W_1, H_1), ..., (W_{L-1}, H_{L-1})]
        Refer to BoxeR: Box-Attention for 2D and 3D Transformers for multi-scale reference box
        """
        nb, nseq = ref_box.shape[:2]
        box_offsets = tF.linear(
            query, self.linear_box_weight, self.linear_box_bias
        )  # nb x nseq x (n_levels x 4)
        box_offsets = box_offsets.view(nb, nseq, -1, 4)  # nb x nseq x n_levels x 4
        norm_expd = torch.cat([box_normalizer, box_normalizer], dim=-1)  # n_levels x 4
        rel_box_offsets = box_offsets / norm_expd.view(
            1, 1, -1, 4
        )  # nb x nseq x n_levels x 4
        ms_box = ref_box + rel_box_offsets
        return ms_box.reshape(nb, nseq, 1, -1, 1, 4).repeat(
            1, 1, self.n_heads, 1, self.n_points, 1
        )

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        ref_boxes,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        Custom addition:
        valid_boxes: N x Length_{query} x n_scale x 4, xyxy box in padding rel for box masked dformable attention
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(
                input_padding_mask[..., None], float(0)
            )  # negative infinity
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = (
            self.sampling_offsets(query)
            .view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            .sigmoid()
        )  # rel in box
        box_offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        ms_ref_boxes = self.box_refiner(
            query, ref_boxes, box_offset_normalizer
        )  # N x Length_{query} x n_head x n_level x n_points x 4
        boxs_wh = torch.stack(
            [
                ms_ref_boxes[..., 2] - ms_ref_boxes[..., 0],
                ms_ref_boxes[..., 3] - ms_ref_boxes[..., 1],
            ],
            dim=-1,
        )
        sampling_locations = sampling_offsets * boxs_wh + ms_ref_boxes[..., :2]

        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = tF.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        if self.r_samp_pt:
            samp_pts = (
                sampling_locations.detach()
                .clamp(min=0, max=1)
                .view(N, Len_q, self.n_heads, -1, 2)
            )
            return (
                samp_pts,
                attention_weights.detach().view(N, Len_q, self.n_heads, -1),
                output,
            )
        return output


class MaskMSDeformAttn(MSDeformAttn):
    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        valid_boxes,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        Custom addition:
        valid_boxes: N x Length_{query} x n_scale x 4, xyxy box in padding rel for box masked dformable attention
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(
                input_padding_mask[..., None], float(0)
            )  # negative infinity
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        valid_boxes_expd = valid_boxes.view(N, Len_q, 1, -1, 1, 4)
        mask_out_x = torch.logical_or(
            sampling_locations[..., 0] < valid_boxes_expd[..., 0],
            sampling_locations[..., 0] > valid_boxes_expd[..., 2],
        )
        mask_out_y = torch.logical_or(
            sampling_locations[..., 1] < valid_boxes_expd[..., 1],
            sampling_locations[..., 1] > valid_boxes_expd[..., 3],
        )
        att_out_mask = torch.logical_or(mask_out_x, mask_out_y)
        attention_weights = attention_weights.masked_fill(
            att_out_mask.view(N, Len_q, self.n_heads, self.n_levels * self.n_points),
            float(-1e10),
        )
        attention_weights = tF.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        if self.r_samp_pt:
            samp_pts = (
                sampling_locations.detach()
                .clamp(min=0, max=1)
                .view(N, Len_q, self.n_heads, -1, 2)
            )
            return (
                samp_pts,
                attention_weights.detach().view(N, Len_q, self.n_heads, -1),
                output,
            )
        return output


class PlainMsDeformAttn(MSDeformAttn):
    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        ref_boxes,
        input_padding_mask=None,
    ):
        return super().forward(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )


class DeformableTransformerDecoderLayerSA(DeformableTransformerDecoderLayer):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        sa_mode="original",
        norm="ln",
        dynamic_dim=64,
        ca_res=True,
        ca_mode="none",
    ):
        super(DeformableTransformerDecoderLayer, self).__init__()
        self.sa = sa_mode
        self.ca_res = ca_res
        norm_type = {"ln": nn.LayerNorm, "bn": CompBN1D}[norm]
        # cross attention
        self.ca_mode = ca_mode
        if ca_mode == "none":
            self.cross_attn = PlainMsDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True
            )
        elif ca_mode == "mask":
            self.cross_attn = MaskMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True
            )
        elif ca_mode == "box":
            self.cross_attn = BoxMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True, box_ref=False
            )
        else:
            self.cross_attn = BoxMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True, box_ref=True
            )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = norm_type(d_model)
        if self.sa == sa_modes[0]:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        else:
            # replace self attention
            if self.sa in sa_modes[1:3]:
                in_proj = nn.Linear(d_model, d_model)
                out_proj = nn.Linear(d_model, d_model)
                nn.init.xavier_normal_(in_proj.weight)
                nn.init.constant_(in_proj.bias, 0)
                nn.init.constant_(out_proj.bias, 0)
                self._sa_trans = nn.Sequential(
                    in_proj,
                    out_proj,
                )
            else:
                assert self.sa in sa_modes[3:]
                self.ddim = dynamic_dim
                self.dparam = nn.Linear(d_model, 2 * d_model * dynamic_dim)
                self.dnorm1 = norm_type(dynamic_dim)
                self.dact1 = nn.ReLU(inplace=True)
                self.dnorm2 = norm_type(d_model)
                self.dact2 = nn.ReLU(inplace=True)
            self.self_attn = (
                self._comp_sa
            )  # nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = norm_type(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = norm_type(d_model)
        self.d_model = d_model

    def _comp_sa(self, trivial_q, trivial_key, value):
        """
        q,k,v: seq x batch x channel
        output: 1 x seq x batch x channel
        """
        if self.sa in sa_modes[1:3]:
            a_values = self._sa_trans(value.permute(1, 0, 2))  # batch x seq x channel
            return a_values.permute(1, 0, 2).unsqueeze(0)  # seq x batch x channel
        else:
            seq, batch = value.shape[:2]
            a_values = value.reshape(-1, 1, self.d_model)  # bq x 1 x dm
            parameters = self.dparam(a_values)  # bq x 1 x d_param
            param1 = parameters[:, :, : self.ddim * self.d_model].view(
                -1, self.d_model, self.ddim
            )  # bq x dm x dd
            param2 = parameters[:, :, self.ddim * self.d_model :].view(
                -1, self.ddim, self.d_model
            )  # bq x dd x dm
            a_values = torch.bmm(a_values, param1)  # bq x 1 x dd
            a_values = self.dnorm1(a_values)
            a_values = self.dact1(a_values)

            a_values = torch.bmm(a_values, param2)
            a_values = self.dnorm2(a_values)
            a_values = self.dact2(a_values)  # bq x 1 x dm
            return a_values.reshape(1, seq, batch, -1)

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        ref_boxes=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(
            0, 1
        )  # batch x seq x channel
        if self.sa == sa_modes[0] or "res" in self.sa:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt = self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        samp_pts, samp_atts, tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            ref_boxes,
            src_padding_mask,
        )
        if self.ca_res:
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return samp_pts, samp_atts, tgt


class DeformableTransformerDecoderLayerPrompt(DeformableTransformerDecoderLayer):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        sa_mode="img-wise",
        norm="ln",
        ca_res=True,
        ca_mode="none",
        out_type="instance",
        prompt_pre=True,
    ):
        super(DeformableTransformerDecoderLayer, self).__init__()
        self.sa = sa_mode
        assert self.sa in sa_prompt
        if self.sa == sa_prompt[0]:
            assert out_type == prompt_out_feat[0]
        self.ca_res = ca_res
        self.out_type = out_type
        self.prompt_pre = prompt_pre
        norm_type = {"ln": nn.LayerNorm, "bn": CompBN1D}[norm]
        # cross attention
        self.ca_mode = ca_mode
        if ca_mode == "none":
            self.cross_attn = PlainMsDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True
            )
        elif ca_mode == "mask":
            self.cross_attn = MaskMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True
            )
        elif ca_mode == "box":
            self.cross_attn = BoxMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True, box_ref=False
            )
        else:
            self.cross_attn = BoxMSDeformAttn(
                d_model, n_levels, n_heads, n_points, r_samp_pt=True, box_ref=True
            )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = norm_type(d_model)
        assert self.sa in sa_prompt
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = norm_type(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = norm_type(d_model)
        self.d_model = d_model

    def forward(
        self,
        tgt,
        query_pos,
        prompts_per_img,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        ref_boxes=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        l_p = prompts_per_img.shape[1]
        b, l, c = tgt.shape
        if self.sa == sa_prompt[0]:
            if self.prompt_pre:
                tgt = torch.cat([prompts_per_img, tgt], dim=1)  # b x l x c
                q = k = torch.cat([prompts_per_img, k], dim=1)
            else:
                tgt = torch.cat([tgt, prompts_per_img], dim=1)  # b x l x c
                q = k = torch.cat([k, prompts_per_img], dim=1)
        else:

            tgt = tgt.view(-1, 1, c)
            prompts = (
                prompts_per_img.unsqueeze(1).expand(b, l, l_p, c).view(-1, l_p, c)
            )  # b x 1 x l_p x c ->  b x l x l_p x c -> bl x l_p x c
            if self.prompt_pre:
                tgt = torch.cat([prompts, tgt], dim=1)  # bl x (l_p+1) x c
                q = k = torch.cat([prompts, k.view(-1, 1, c)], dim=1)
            else:
                tgt = torch.cat([tgt, prompts], dim=1)
                q = k = torch.cat([k.view(-1, 1, c), prompts], dim=1)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(
            0, 1
        )  # batch x seq x channel
        tgt = tgt + self.dropout2(tgt2)

        tgt = self.norm2(tgt)
        if self.sa == sa_prompt[0]:
            if self.prompt_pre:
                tgt = tgt[:, l_p:]
            else:
                tgt = tgt[:, :-l_p]
        else:
            if self.out_type == prompt_out_feat[0]:
                if self.prompt_pre:
                    tgt = tgt[:, l_p:].view(b, l, 1, c).squeeze(-2)
                else:
                    tgt = tgt[:, :-l_p].view(b, l, 1, c).squeeze(-2)
            elif self.out_type == prompt_out_feat[1]:
                if self.prompt_pre:
                    tgt = tgt[:, :l_p].mean(1).view(b, l, c)
                else:
                    tgt = tgt[:, -l_p:].mean(1).view(b, l, c)
            else:
                tgt = tgt.mean(1).view(b, l, c)

        # cross attention
        samp_pts, samp_atts, tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            ref_boxes,
            src_padding_mask,
        )
        if self.ca_res:
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return samp_pts, samp_atts, tgt


def get_bt_neck(type_name, in_dim):
    if type_name == "none":
        return nn.Identity()
    else:
        bt_neck = {"bn": CompBN1D, "ln": nn.LayerNorm}[type_name](in_dim)
        bt_neck.bias.requires_grad_(False)
        nn.init.constant_(bt_neck.weight, 1.0)
        nn.init.constant_(bt_neck.bias, 0.0)
        return bt_neck


class DTransDecoderHead(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        dec_sa = head_cfg.SA
        self.dec_norm = head_cfg.NORM
        dec_ca_res = head_cfg.CA_RES
        ca_mode = head_cfg.CA_MODE
        dec_layer = DeformableTransformerDecoderLayerSA(
            d_model=head_cfg.PERSON_FEATURE.DIM,
            d_ffn=head_cfg.DIM_FEEDFORWARD,
            dropout=head_cfg.DROPOUT,
            n_levels=head_cfg.N_FEATURE_LEVELS,
            n_heads=head_cfg.N_HEADS,
            n_points=head_cfg.N_POINTS,
            sa_mode=dec_sa,
            norm=self.dec_norm,
            ca_res=dec_ca_res,
            ca_mode=ca_mode,
        )
        self.dec_layers = _get_clones(dec_layer, head_cfg.N_LAYERS)
        self.query_gradient = head_cfg.QUERY_BP
        self.pos_gradient = head_cfg.POS_BP
        self.vis_period = head_cfg.VIS_PERIOD
        self.bottelnecks = nn.ModuleList()
        bt_neck_type = head_cfg.BOTTELNECK
        for li in range(head_cfg.LOSS.AUX_LOSS + 1):
            self.bottelnecks.append(get_bt_neck(bt_neck_type, self.pfeat_dim))
        self.multi_layer_query = head_cfg.MULTI_LAYER_QUERY

    def get_pfeats(
        self,
        enc_memory,
        det_outputs,
        targets,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        pred_boxes = det_outputs["pred_boxes"]  # b x n x 4 xyxy_abs
        if targets is not None:
            aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
        else:
            aug_whwh_bs = img_aug_whwh.unsqueeze(1)
        pred_boxes_ccwh_rel = box_xyxy_to_cxcywh(
            (pred_boxes / aug_whwh_bs).flatten(0, 1)
        ).reshape(*pred_boxes.shape)
        sample_area = pred_boxes / aug_whwh_bs  # b x n x 4
        sample_area = sample_area[:, :, None] * torch.cat(
            [pts_valid_ratios[:, None]] * 2, dim=-1
        )  # # b x n x scales x 4 in padding
        if self.box_gradient:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2]
        else:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2].detach()  # in aug
        assert ref_pts.shape[-1] == 2
        ref_pts_input = (
            ref_pts[:, :, None] * pts_valid_ratios[:, None]
        )  # b x n x l x 2 in padding
        if query_pos is not None and not self.pos_gradient:
            query_pos = query_pos.detach()
        if not self.query_gradient:
            queries = queries.detach()
        output = queries
        intermediate_out = []
        intermediate_pts = []
        intermediate_atts = []
        for layer in self.dec_layers:
            samp_pts, samp_atts, output = layer(
                output,
                query_pos,
                ref_pts_input,
                enc_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                ref_boxes=sample_area,
            )  # b x n x p
            intermediate_out.append(output)
            intermediate_pts.append(samp_pts)
            intermediate_atts.append(samp_atts)
        return (
            ref_pts,  # in aug
            intermediate_pts,  # in padding
            intermediate_atts,
            intermediate_out,
        )

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]
            to_head_query_pos = torch.cat([query_pos] * len(vals), dim=1)
        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
            to_head_query_pos = query_pos
        ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
            enc_memory,
            to_head_det_outputs,
            targets,
            to_head_queries,
            to_head_query_pos,
            pts_valid_ratios,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_mask_flatten,
            img_aug_whwh,
        )
        len_pfeats = len(pfeats_out)
        head_outputs = {}
        if self.training:
            # NOTE update to only involve boxes with valid ids
            head_outputs["losses"] = {}
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )

            n_losses = len(self.loss_layers)
            if n_losses > 1:  # supervise multi layers in this head
                # TODO compute only with ids
                head_aux_outputs = []

                for i in range(n_losses):
                    if i < n_losses - 1:
                        head_aux_outputs.append(
                            {"assign_ids": assign_ids[:, :n_queries]}
                        )
                    else:
                        head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                aux_losses = {}
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    pfeats_cls = self.bottelnecks[i](pfeats_out[i + loss_i_offset])
                    losses = self.compute_losses_lvl(
                        pfeats_cls,
                        assign_ids,
                        as_s,
                        to_head_det_outputs["pred_logits"],
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            pfeats_out[i + loss_i_offset],
                            assign_ids,
                            as_s,
                            to_head_det_outputs["pred_logits"],
                        )
                        losses.update(metric_losses)
                    if i < n_losses - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
                pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    assign_ids,
                    as_s,
                    to_head_det_outputs["pred_logits"],
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_out[-1],
                        assign_ids,
                        as_s,
                        to_head_det_outputs["pred_logits"],
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.d_points_vis(
                    det_outputs["pred_logits"].squeeze(2).sigmoid(),
                    ref_pts[:, :n_queries],
                    [spts[:, :n_queries] for spts in samp_pts],
                    [satts[:, :n_queries] for satts in samp_atts],
                    targets,
                )
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs

    @torch.no_grad()
    def d_points_vis(
        self,
        det_scores,
        refs,
        head_pts,
        head_atts,
        targets,
        score_threds=[0.2, 0.5, 0.7],
    ):
        img_norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img_norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        trans_t2img_rgb_t = lambda t: (t * img_norm_std + img_norm_mean) * 255.0
        storage = get_event_storage()
        lyn = len(head_atts)
        if isinstance(head_atts[0], torch.Tensor):
            bn, qn, hn, pn = head_atts[0].shape
        else:
            bn = len(head_atts[0])
            hn = head_atts[0][0].shape[2]
        # generate colors
        colors_tps = []
        for i in np.arange(0.0, 360.0, 360.0 / hn):
            hue = i / 360.0
            saturation = 0.9
            colors_tps.append((hue, saturation))
        # colors_np = torch.tensor(colors, dtype=torch.float32)  # * 255.0, matplotlib accept (0,1)
        def get_head_color(hi, att):
            h_clr_tp = colors_tps[hi]
            lightness = 1 - att if att > 1e-8 else 0.0  # 1 - att
            return colorsys.hls_to_rgb(h_clr_tp[0], lightness, h_clr_tp[1])

        for bi in range(bn):
            img_rgb_t = trans_t2img_rgb_t(targets[bi]["image_t"].cpu())  # 3 x h x w
            img_rgb = img_rgb_t.permute(1, 2, 0).numpy()
            bi_scores = det_scores[bi]
            s_t = score_threds + [1]
            vis_s_imgs = []
            for ti in range(len(s_t) - 1):
                vmask = torch.logical_and(bi_scores >= s_t[ti], bi_scores < s_t[ti + 1])
                # v_refs=refs[bi][vmask]
                ly_is = [0, -1] if lyn > 1 else [0]
                vis_ly_imgs = []
                for lyi in ly_is:
                    visualize_samp = Visualizer(img_rgb.copy())
                    v_atts = head_atts[lyi][bi][vmask].cpu().numpy()
                    v_pts = head_pts[lyi][bi][vmask].cpu().numpy()
                    ql = v_atts.shape[0]
                    for qi in range(ql):
                        for hi, (h_pts, h_atts) in enumerate(
                            zip(v_pts[qi], v_atts[qi])
                        ):
                            for pt, att in zip(h_pts, h_atts):
                                clr = get_head_color(hi, att)
                                pt_abs = pt * np.array(
                                    [img_rgb.shape[1], img_rgb.shape[0]]
                                )
                                visualize_samp.draw_circle(
                                    pt_abs.tolist(),
                                    color=clr,
                                    radius=2,
                                )
                    vis_img = visualize_samp.get_output().get_image()
                    vis_ly_imgs.append(vis_img)
                vis_ly_img = np.concatenate(vis_ly_imgs, axis=0)
                vis_s_imgs.append(vis_ly_img)
            vis_s_img = np.concatenate(vis_s_imgs, axis=1)
            vis_img_t = tvF.to_tensor(vis_s_img)
            storage.put_image("img_{}/ca_samps".format(bi), vis_img_t)


class DTransDecoderHeadClean(DTransDecoderHead):
    def split_id_asc_ptfeat_logits(self, ids, as_s, det_outputs, det_feats):
        bn_ids = []
        bn_asc = []
        bn_ptfeats = []
        bn_logits = []
        bn = ids.shape[0]
        num_pfeats = 0
        for bi in range(bn):
            mask = ids[bi] > -2
            bn_ids.append(ids[bi][mask])
            bn_asc.append(as_s[bi][mask] if as_s is not None else None)
            keep_pfeats = det_feats[bi][mask]
            keep_pfeats = keep_pfeats.view(-1, keep_pfeats.shape[-1])
            bn_ptfeats.append(keep_pfeats)
            num_pfeats += keep_pfeats.shape[0]
            bn_logits.append(det_outputs["pred_logits"][bi][mask])
        while num_pfeats < 2:
            # compatible with bn and dist
            bi = random.randint(0, bn - 1)
            li = random.randint(0, det_feats.shape[1] - 1)
            bi_ptfeats = bn_ptfeats[bi]
            bn_ptfeats[bi] = torch.cat([bi_ptfeats, det_feats[bi][li : li + 1]], dim=0)
            num_pfeats += 1
            bi_ids = bn_ids[bi]
            bn_ids[bi] = torch.cat([bi_ids, ids[bi][li : li + 1]], dim=0)
            bi_asc = bn_asc[bi]
            if bi_asc is not None:
                bn_asc[bi] = torch.cat([bi_asc, as_s[bi][li : li + 1]], dim=0)
            bi_logits = bn_logits[bi]
            bn_logits[bi] = torch.cat(
                [bi_logits, det_outputs["pred_logits"][bi][li : li + 1]], dim=0
            )

        return bn_ids, bn_asc, bn_ptfeats, bn_logits

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # NOTE aux share not kept
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]
            to_head_query_pos = torch.cat([query_pos] * len(vals), dim=1)
        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
            to_head_query_pos = query_pos
        ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
            enc_memory,
            to_head_det_outputs,
            targets,
            to_head_queries,
            to_head_query_pos,
            pts_valid_ratios,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_mask_flatten,
            img_aug_whwh,
        )
        head_outputs = {}
        len_pfeats = len(pfeats_out)
        if self.training:
            head_outputs["losses"] = {}
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
            n_losses = self.out_lvls
            if n_losses > 1:
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s, lvl_pfeats, lvl_logits = [], [], [], []
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    (
                        bn_ids,
                        bn_asc,
                        bn_ptfeats,
                        bn_logits,
                    ) = self.split_id_asc_ptfeat_logits(
                        assign_ids,
                        as_s,
                        to_head_det_outputs,
                        pfeats_out[i + loss_i_offset],
                    )
                    lvl_assign_ids.append(bn_ids)
                    lvl_as_s.append(bn_asc)
                    lvl_pfeats.append(bn_ptfeats)
                    lvl_logits.append(bn_logits)
                    if i < n_losses - 1:
                        head_aux_outputs.append(
                            {"assign_ids": assign_ids[:, :n_queries]}
                        )
                    else:
                        head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                aux_losses = {}
                for i in range(n_losses):
                    li_pfeats_mtc = torch.cat(lvl_pfeats[i], dim=0)
                    li_pfeats_cls = self.bottelnecks[i](li_pfeats_mtc)
                    li_assign_ids = torch.cat(lvl_assign_ids[i])
                    li_as_s = (
                        torch.cat(lvl_as_s[i]) if lvl_as_s[i][0] is not None else None
                    )
                    li_logits = torch.cat(lvl_logits[i])
                    losses = self.compute_losses_lvl(
                        li_pfeats_cls,
                        li_assign_ids,
                        li_as_s,
                        li_logits,
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            li_pfeats_mtc,
                            li_assign_ids,
                            li_as_s,
                            li_logits,
                        )
                        losses.update(metric_losses)
                    if i < n_losses - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            else:
                (
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    assign_ids,
                    as_s,
                    to_head_det_outputs,
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bottelnecks[-1](pfeats_mtc)
                valid_ids = torch.cat(bn_ids)
                valid_as_s = torch.cat(bn_asc) if bn_asc[0] is not None else None
                valid_logits = torch.cat(bn_logits)
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    valid_ids,
                    valid_as_s,
                    valid_logits,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_mtc,
                        valid_ids,
                        valid_as_s,
                        valid_logits,
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.d_points_vis(
                    det_outputs["pred_logits"].squeeze(2).sigmoid(),
                    ref_pts[:, :n_queries],
                    [spts[:, :n_queries] for spts in samp_pts],
                    [satts[:, :n_queries] for satts in samp_atts],
                    targets,
                )
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs


class MscDTransDecoderHead(DTransDecoderHead):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        dec_sa = head_cfg.SA
        self.dec_norm = head_cfg.NORM
        dec_ca_res = head_cfg.CA_RES
        ca_mode = head_cfg.CA_MODE
        dec_layer = DeformableTransformerDecoderLayerSA(
            d_model=head_cfg.PERSON_FEATURE.DIM,
            d_ffn=head_cfg.DIM_FEEDFORWARD,
            dropout=head_cfg.DROPOUT,
            n_levels=1,
            n_heads=head_cfg.N_HEADS,
            n_points=head_cfg.N_POINTS,
            sa_mode=dec_sa,
            norm=self.dec_norm,
            ca_res=dec_ca_res,
            ca_mode=ca_mode,
        )
        self.n_levels = head_cfg.N_FEATURE_LEVELS
        self.n_layers = head_cfg.N_LAYERS
        self.dec_layers = _get_clones(
            _get_clones(dec_layer, self.n_layers), self.n_levels
        )  # n_level x n_layer
        self.query_gradient = head_cfg.QUERY_BP
        self.pos_gradient = head_cfg.POS_BP
        self.vis_period = head_cfg.VIS_PERIOD
        self.fuse = self._build_msc_fuse(head_cfg.FUSE)(self.pfeat_dim, self.pfeat_dim)
        self.bottelnecks = nn.ModuleList()
        bt_neck_type = head_cfg.BOTTELNECK
        for li in range(len(self.loss_layers)):
            self.bottelnecks.append(get_bt_neck(bt_neck_type, self.pfeat_dim))

    def _build_msc_fuse(self, fuse_type):
        if fuse_type == "none":
            return nn.Identity
        elif fuse_type == "linear":
            return nn.Linear
        else:
            raise ValueError("Unsupported fusion type!")

    def get_pfeats(
        self,
        enc_memory,
        det_outputs,
        targets,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        pred_boxes = det_outputs["pred_boxes"]  # b x n x 4 xyxy_abs
        if targets is not None:
            aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
        else:
            aug_whwh_bs = img_aug_whwh.unsqueeze(1)
        pred_boxes_ccwh_rel = box_xyxy_to_cxcywh(
            (pred_boxes / aug_whwh_bs).flatten(0, 1)
        ).reshape(*pred_boxes.shape)
        sample_area = pred_boxes / aug_whwh_bs  # b x n x 4
        sample_area = sample_area[:, :, None] * torch.cat(
            [pts_valid_ratios[:, None]] * 2, dim=-1
        )  # # b x n x scales x 4 in padding
        if self.box_gradient:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2]
        else:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2].detach()  # in aug
        assert ref_pts.shape[-1] == 2
        ref_pts_input = (
            ref_pts[:, :, None] * pts_valid_ratios[:, None]
        )  # b x n x l x 2 in padding
        if not self.pos_gradient:
            query_pos = query_pos.detach()
        if not self.query_gradient:
            queries = queries.detach()

        lvls_out = []
        lvls_pts = []
        lvls_atts = []
        memory_level_sizes = (
            torch.cat(memory_level_start_index[1:], enc_memory.shape[1:2])
            - memory_level_start_index
        )
        enc_memory_lvls = torch.split(enc_memory, memory_level_sizes, dim=1)
        mem_mask_lvls = torch.split(memory_mask_flatten, memory_level_sizes, dim=1)
        for lvl_i, (lvl_layers, lvl_enc_mem, lvl_mem_mask) in enumerate(
            zip(self.dec_layers, enc_memory_lvls, mem_mask_lvls)
        ):
            inter_out = []
            inter_pts = []
            inter_atts = []
            output = queries
            for layer in lvl_layers:
                samp_pts, samp_atts, output = layer(
                    output,
                    query_pos,
                    ref_pts_input[..., lvl_i : lvl_i + 1, :],
                    lvl_enc_mem,
                    memory_spatial_shapes[lvl_i : lvl_i + 1],
                    memory_level_start_index[0:1],
                    lvl_mem_mask,
                    ref_boxes=sample_area[..., lvl_i : lvl_i + 1, :],
                )  # b x n x p
                inter_out.append(output)
                inter_pts.append(samp_pts)
                inter_atts.append(samp_atts)
            lvls_out.append(inter_out)
            lvls_pts.append(inter_pts)
            lvls_atts.append(inter_atts)
        intermediate_pts = [
            torch.cat([li_pts[di] for li_pts in lvls_pts], dim=-2)
            for di in range(self.n_layers)
        ]
        intermediate_atts = [
            torch.cat([li_atts[di] for li_atts in lvls_atts], dim=-1)
            for di in range(self.n_layers)
        ]
        intermediate_out = [
            self.fuse(torch.cat([li_out[di] for li_out in lvls_out], dim=-1))
            for di in range(self.n_layers)
        ]
        return (
            ref_pts,  # in aug
            intermediate_pts,  # in padding
            intermediate_atts,
            intermediate_out,
        )


class MscDTransDecoderHeadClean(MscDTransDecoderHead):
    def split_id_asc_ptfeat_logits(self, ids, as_s, det_outputs, det_feats):
        bn_ids = []
        bn_asc = []
        bn_ptfeats = []
        bn_logits = []
        bn = ids.shape[0]
        num_pfeats = 0
        for bi in range(bn):
            mask = ids[bi] > -2
            bn_ids.append(ids[bi][mask])
            bn_asc.append(as_s[bi][mask] if as_s is not None else None)
            keep_pfeats = det_feats[bi][mask]
            keep_pfeats = keep_pfeats.view(-1, keep_pfeats.shape[-1])
            bn_ptfeats.append(keep_pfeats)
            num_pfeats += keep_pfeats.shape[0]
            bn_logits.append(det_outputs["pred_logits"][bi][mask])
        while num_pfeats < 2:
            # compatible with bn and dist
            bi = random.randint(0, bn - 1)
            li = random.randint(0, det_feats.shape[1] - 1)
            bi_ptfeats = bn_ptfeats[bi]
            bn_ptfeats[bi] = torch.cat([bi_ptfeats, det_feats[bi][li : li + 1]], dim=0)
            num_pfeats += 1
            bi_ids = bn_ids[bi]
            bn_ids[bi] = torch.cat([bi_ids, ids[bi][li : li + 1]], dim=0)
            bi_asc = bn_asc[bi]
            if bi_asc is not None:
                bn_asc[bi] = torch.cat([bi_asc, as_s[bi][li : li + 1]], dim=0)
            bi_logits = bn_logits[bi]
            bn_logits[bi] = torch.cat(
                [bi_logits, det_outputs["pred_logits"][bi][li : li + 1]], dim=0
            )

        return bn_ids, bn_asc, bn_ptfeats, bn_logits

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # NOTE aux share not kept
        ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
            enc_memory,
            det_outputs,
            targets,
            queries,
            query_pos,
            pts_valid_ratios,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_mask_flatten,
            img_aug_whwh,
        )
        head_outputs = {}
        len_pfeats = len(pfeats_out)
        if self.training:
            head_outputs["losses"] = {}
            assign_ids, as_s = self._id_assign_method(
                det_outputs, targets, det_match_indices[-1]
            )
            n_losses = len(self.loss_layers)
            if n_losses > 1:
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s, lvl_pfeats, lvl_logits = [], [], [], []
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    (
                        bn_ids,
                        bn_asc,
                        bn_ptfeats,
                        bn_logits,
                    ) = self.split_id_asc_ptfeat_logits(
                        assign_ids,
                        as_s,
                        det_outputs,
                        pfeats_out[i + loss_i_offset],
                    )
                    lvl_assign_ids.append(bn_ids)
                    lvl_as_s.append(bn_asc)
                    lvl_pfeats.append(bn_ptfeats)
                    lvl_logits.append(bn_logits)
                    if i < n_losses - 1:
                        head_aux_outputs.append({"assign_ids": assign_ids})
                    else:
                        head_outputs["assign_ids"] = assign_ids
                aux_losses = {}
                for i in range(n_losses):
                    li_pfeats_mtc = torch.cat(lvl_pfeats[i], dim=0)
                    li_pfeats_cls = self.bottelnecks[i](li_pfeats_mtc)
                    li_assign_ids = torch.cat(lvl_assign_ids[i])
                    li_as_s = (
                        torch.cat(lvl_as_s[i]) if lvl_as_s[i][0] is not None else None
                    )
                    li_logits = torch.cat(lvl_logits[i])
                    losses = self.compute_losses_lvl(
                        li_pfeats_cls,
                        li_assign_ids,
                        li_as_s,
                        li_logits,
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            li_pfeats_mtc,
                            li_assign_ids,
                            li_as_s,
                            li_logits,
                        )
                        losses.update(metric_losses)
                    if i < n_losses - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            else:
                (
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    assign_ids,
                    as_s,
                    det_outputs,
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bottelnecks[-1](pfeats_mtc)
                valid_ids = torch.cat(bn_ids)
                valid_as_s = torch.cat(bn_asc) if bn_asc[0] is not None else None
                valid_logits = torch.cat(bn_logits)
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    valid_ids,
                    valid_as_s,
                    valid_logits,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_mtc,
                        valid_ids,
                        valid_as_s,
                        valid_logits,
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.d_points_vis(
                    det_outputs["pred_logits"].squeeze(2).sigmoid(),
                    ref_pts,
                    samp_pts,
                    samp_atts,
                    targets,
                )
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs


class DTransDecoderHeadL2P(ReidHeadBase):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        dec_sa = head_cfg.SA
        self.dec_norm = head_cfg.NORM
        dec_ca_res = head_cfg.CA_RES
        ca_mode = head_cfg.CA_MODE
        prompt_cfg = head_cfg.L2P
        dec_layer = DeformableTransformerDecoderLayerPrompt(
            d_model=head_cfg.PERSON_FEATURE.DIM,
            d_ffn=head_cfg.DIM_FEEDFORWARD,
            dropout=head_cfg.DROPOUT,
            n_levels=head_cfg.N_FEATURE_LEVELS,
            n_heads=head_cfg.N_HEADS,
            n_points=head_cfg.N_POINTS,
            sa_mode=dec_sa,
            norm=self.dec_norm,
            ca_res=dec_ca_res,
            ca_mode=ca_mode,
            out_type=prompt_cfg.OUT_TYPE,
            prompt_pre=prompt_cfg.PREPEND,
        )
        self.dec_layers = _get_clones(dec_layer, head_cfg.N_LAYERS)
        self.query_gradient = head_cfg.QUERY_BP
        self.pos_gradient = head_cfg.POS_BP
        self.vis_period = head_cfg.VIS_PERIOD
        self.bottelnecks = nn.ModuleList()
        bt_neck_type = head_cfg.BOTTELNECK
        for li in range(len(self.loss_layers)):
            self.bottelnecks.append(get_bt_neck(bt_neck_type, self.pfeat_dim))
        self.multi_layer_query = head_cfg.MULTI_LAYER_QUERY

        self.prompt_layer_idx = prompt_cfg.LAYERS

        self.prompt_pool = nn.Parameter(
            torch.randn(
                len(self.prompt_layer_idx),
                prompt_cfg.POOL_SIZE,
                prompt_cfg.LENGTH,
                self.pfeat_dim,
            )
        )
        uniform_(self.prompt_pool)
        self.prompt_keys = nn.Parameter(
            torch.randn(prompt_cfg.POOL_SIZE, self.pfeat_dim)
        )
        uniform_(self.prompt_keys)
        self.register_buffer(
            "prompt_select_times", torch.zeros(prompt_cfg.POOL_SIZE, dtype=torch.int32)
        )
        self.register_buffer("task_id", torch.tensor([0], dtype=torch.int32))
        self.register_buffer("total_select", torch.tensor([0], dtype=torch.float32))
        self.register_buffer(
            "prompt_select_freq", torch.zeros(prompt_cfg.POOL_SIZE, dtype=torch.float32)
        )
        self.prompt_topk = prompt_cfg.TOP_K
        self.incoming_task_id = prompt_cfg.TASK_ID
        self.l2p_loss_weight = head_cfg.LOSS.LOSS_WEIGHTS.L2P
        self.prompt_diversify = prompt_cfg.DIVERSIFY
        self.prompt_sync_selection = prompt_cfg.SYNC

    def _param_setup(self):
        """
        Call after loading state
        """
        prev_task_id = self.task_id.item()
        if self.incoming_task_id > prev_task_id:
            # TODO ???
            prev_p_start = prev_task_id * self.prompt_topk
            prev_p_end = (prev_task_id + 1) * self.prompt_topk
            incoming_p_start = self.incoming_task_id * self.prompt_topk
            incoming_p_end = (self.incoming_task_id + 1) * self.prompt_topk
            with torch.no_grad():
                self.prompt_pool[:, incoming_p_start:incoming_p_end].copy_(
                    self.prompt_pool[:, prev_p_start:prev_p_end].clone()
                )
            self.task_id[0] = self.incoming_task_id
        self.prompt_select_freq = self.prompt_select_times / self.total_select

    def get_pfeats(
        self,
        enc_memory,
        det_outputs,
        targets,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        pred_boxes = det_outputs["pred_boxes"]  # b x n x 4 xyxy_abs
        if targets is not None:
            aug_whwh_bs = torch.stack([v["aug_whwh"] for v in targets])
        else:
            aug_whwh_bs = img_aug_whwh.unsqueeze(1)
        pred_boxes_ccwh_rel = box_xyxy_to_cxcywh(
            (pred_boxes / aug_whwh_bs).flatten(0, 1)
        ).reshape(*pred_boxes.shape)
        sample_area = pred_boxes / aug_whwh_bs  # b x n x 4
        sample_area = sample_area[:, :, None] * torch.cat(
            [pts_valid_ratios[:, None]] * 2, dim=-1
        )  # # b x n x scales x 4 in padding
        if self.box_gradient:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2]
        else:
            ref_pts = pred_boxes_ccwh_rel[:, :, :2].detach()  # in aug
        assert ref_pts.shape[-1] == 2
        ref_pts_input = (
            ref_pts[:, :, None] * pts_valid_ratios[:, None]
        )  # b x n x l x 2 in padding
        if not self.pos_gradient:
            query_pos = query_pos.detach()
        if not self.query_gradient:
            queries = queries.detach()
        if self.training:
            prompts, sim_pull = self.prompt_select(queries)
        else:
            prompts = self.prompt_select(queries)
        output = queries
        intermediate_out = []
        intermediate_pts = []
        intermediate_atts = []
        for li, layer in enumerate(self.dec_layers):
            # NOTE selective layer not supported for now
            samp_pts, samp_atts, output = layer(
                output,
                query_pos,
                prompts[li],
                ref_pts_input,
                enc_memory,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                ref_boxes=sample_area,
            )  # b x n x p
            intermediate_out.append(output)
            intermediate_pts.append(samp_pts)
            intermediate_atts.append(samp_atts)
        if not self.training:
            return (
                ref_pts,  # in aug
                intermediate_pts,  # in padding
                intermediate_atts,
                intermediate_out,
            )
        else:
            return (
                ref_pts,  # in aug
                intermediate_pts,  # in padding
                intermediate_atts,
                intermediate_out,
                sim_pull,
            )

    def prompt_select(self, queries):
        """
        queries: b x l x c
        """
        query_norm = tF.normalize(queries, dim=-1).flatten(0, 1)  # bl x c
        key_norm = tF.normalize(self.prompt_keys, dim=-1)  # lp x c
        cos_sim = torch.matmul(query_norm, key_norm.transpose(0, 1))  # bl x lp
        if self.prompt_sync_selection:
            cos_sims = comm.all_gather(cos_sim)
            all_sim_mats = []
            for sim_mat in cos_sims:
                all_sim_mats.append(sim_mat.to(self.prompt_pool.device))
            cos_sim = torch.cat(all_sim_mats, dim=0)
        if self.prompt_diversify and self.training and self.task_id.item() > 0:
            factor = (1 - self.prompt_select_freq).unsqueeze(0)
            cos_sim = cos_sim * factor
        sim_top_k, topk_idx_inst = torch.topk(
            cos_sim, k=self.prompt_topk, dim=1
        )  # bl x topk
        prompt_id, id_counts = torch.unique(topk_idx_inst, return_counts=True)
        _, major_idx = torch.topk(id_counts, self.prompt_topk)
        major_prompt_id = prompt_id[major_idx]
        selected_prompts = self.prompt_pool[:, major_prompt_id]  # layer x topk x lp x c
        batch_selected_prompts = (
            selected_prompts.unsqueeze(1)
            .flatten(2, 3)
            .expand(-1, queries.shape[0], -1, -1)
        )  # layer x b x p_l x c
        if self.training:
            batch_selected_keys = (
                key_norm[major_prompt_id]
                .unsqueeze(0)
                .expand(query_norm.shape[0], -1, -1)
            )
            query_norm = query_norm.unsqueeze(1)
            sim_pull = (batch_selected_keys * query_norm).sum() / query_norm.shape[0]
            with torch.no_grad():
                self.prompt_select_times[major_prompt_id] += 1
                self.total_select += 1
            return batch_selected_prompts, sim_pull
        else:
            return batch_selected_prompts

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]
            to_head_query_pos = torch.cat([query_pos] * len(vals), dim=1)
        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
            to_head_query_pos = query_pos
        if self.training:
            ref_pts, samp_pts, samp_atts, pfeats_out, sim_pull = self.get_pfeats(
                enc_memory,
                to_head_det_outputs,
                targets,
                to_head_queries,
                to_head_query_pos,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
        else:
            ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
                enc_memory,
                to_head_det_outputs,
                targets,
                to_head_queries,
                to_head_query_pos,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
        len_pfeats = len(pfeats_out)
        head_outputs = {}
        if self.training:
            # NOTE update to only involve boxes with valid ids
            head_outputs["losses"] = {"sim_l2p": -self.l2p_loss_weight * sim_pull}
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )

            n_losses = len(self.loss_layers)
            if n_losses > 1:  # supervise multi layers in this head
                # TODO compute only with ids
                head_aux_outputs = []

                for i in range(n_losses):
                    if i < n_losses - 1:
                        head_aux_outputs.append(
                            {"assign_ids": assign_ids[:, :n_queries]}
                        )
                    else:
                        head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                aux_losses = {}
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    pfeats_cls = self.bottelnecks[i](pfeats_out[i + loss_i_offset])
                    losses = self.compute_losses_lvl(
                        pfeats_cls,
                        assign_ids,
                        as_s,
                        to_head_det_outputs["pred_logits"],
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            pfeats_out[i + loss_i_offset],
                            assign_ids,
                            as_s,
                            to_head_det_outputs["pred_logits"],
                        )
                        losses.update(metric_losses)
                    if i < n_losses - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs
            else:
                pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    assign_ids,
                    as_s,
                    to_head_det_outputs["pred_logits"],
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_out[-1],
                        assign_ids,
                        as_s,
                        to_head_det_outputs["pred_logits"],
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.d_points_vis(
                    det_outputs["pred_logits"].squeeze(2).sigmoid(),
                    ref_pts[:, :n_queries],
                    [spts[:, :n_queries] for spts in samp_pts],
                    [satts[:, :n_queries] for satts in samp_atts],
                    targets,
                )
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs

    @torch.no_grad()
    def d_points_vis(
        self,
        det_scores,
        refs,
        head_pts,
        head_atts,
        targets,
        score_threds=[0.2, 0.5, 0.7],
    ):
        img_norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img_norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        trans_t2img_rgb_t = lambda t: (t * img_norm_std + img_norm_mean) * 255.0
        storage = get_event_storage()
        lyn = len(head_atts)
        if isinstance(head_atts[0], torch.Tensor):
            bn, qn, hn, pn = head_atts[0].shape
        else:
            bn = len(head_atts[0])
            hn = head_atts[0][0].shape[2]
        # generate colors
        colors_tps = []
        for i in np.arange(0.0, 360.0, 360.0 / hn):
            hue = i / 360.0
            saturation = 0.9
            colors_tps.append((hue, saturation))
        # colors_np = torch.tensor(colors, dtype=torch.float32)  # * 255.0, matplotlib accept (0,1)
        def get_head_color(hi, att):
            h_clr_tp = colors_tps[hi]
            lightness = 1 - att if att > 1e-8 else 0.0  # 1 - att
            return colorsys.hls_to_rgb(h_clr_tp[0], lightness, h_clr_tp[1])

        for bi in range(bn):
            img_rgb_t = trans_t2img_rgb_t(targets[bi]["image_t"].cpu())  # 3 x h x w
            img_rgb = img_rgb_t.permute(1, 2, 0).numpy()
            bi_scores = det_scores[bi]
            s_t = score_threds + [1]
            vis_s_imgs = []
            for ti in range(len(s_t) - 1):
                vmask = torch.logical_and(bi_scores >= s_t[ti], bi_scores < s_t[ti + 1])
                # v_refs=refs[bi][vmask]
                ly_is = [0, -1] if lyn > 1 else [0]
                vis_ly_imgs = []
                for lyi in ly_is:
                    visualize_samp = Visualizer(img_rgb.copy())
                    v_atts = head_atts[lyi][bi][vmask].cpu().numpy()
                    v_pts = head_pts[lyi][bi][vmask].cpu().numpy()
                    ql = v_atts.shape[0]
                    for qi in range(ql):
                        for hi, (h_pts, h_atts) in enumerate(
                            zip(v_pts[qi], v_atts[qi])
                        ):
                            for pt, att in zip(h_pts, h_atts):
                                clr = get_head_color(hi, att)
                                pt_abs = pt * np.array(
                                    [img_rgb.shape[1], img_rgb.shape[0]]
                                )
                                visualize_samp.draw_circle(
                                    pt_abs.tolist(),
                                    color=clr,
                                    radius=2,
                                )
                    vis_img = visualize_samp.get_output().get_image()
                    vis_ly_imgs.append(vis_img)
                vis_ly_img = np.concatenate(vis_ly_imgs, axis=0)
                vis_s_imgs.append(vis_ly_img)
            vis_s_img = np.concatenate(vis_s_imgs, axis=1)
            vis_img_t = tvF.to_tensor(vis_s_img)
            storage.put_image("img_{}/ca_samps".format(bi), vis_img_t)


class DTransDecoderHeadL2PClean(DTransDecoderHeadL2P):
    def split_id_asc_ptfeat_logits(self, ids, as_s, det_outputs, det_feats):
        bn_ids = []
        bn_asc = []
        bn_ptfeats = []
        bn_logits = []
        bn = ids.shape[0]
        num_pfeats = 0
        for bi in range(bn):
            mask = ids[bi] > -2
            bn_ids.append(ids[bi][mask])
            bn_asc.append(as_s[bi][mask] if as_s is not None else None)
            keep_pfeats = det_feats[bi][mask]
            keep_pfeats = keep_pfeats.view(-1, keep_pfeats.shape[-1])
            bn_ptfeats.append(keep_pfeats)
            num_pfeats += keep_pfeats.shape[0]
            bn_logits.append(det_outputs["pred_logits"][bi][mask])
        while num_pfeats < 2:
            # compatible with bn and dist
            bi = random.randint(0, bn - 1)
            li = random.randint(0, det_feats.shape[1] - 1)
            bi_ptfeats = bn_ptfeats[bi]
            bn_ptfeats[bi] = torch.cat([bi_ptfeats, det_feats[bi][li : li + 1]], dim=0)
            num_pfeats += 1
            bi_ids = bn_ids[bi]
            bn_ids[bi] = torch.cat([bi_ids, ids[bi][li : li + 1]], dim=0)
            bi_asc = bn_asc[bi]
            if bi_asc is not None:
                bn_asc[bi] = torch.cat([bi_asc, as_s[bi][li : li + 1]], dim=0)
            bi_logits = bn_logits[bi]
            bn_logits[bi] = torch.cat(
                [bi_logits, det_outputs["pred_logits"][bi][li : li + 1]], dim=0
            )

        return bn_ids, bn_asc, bn_ptfeats, bn_logits

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # NOTE aux share not kept
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]
            to_head_query_pos = torch.cat([query_pos] * len(vals), dim=1)
        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
            to_head_query_pos = query_pos
        if self.training:
            ref_pts, samp_pts, samp_atts, pfeats_out, sim_pull = self.get_pfeats(
                enc_memory,
                to_head_det_outputs,
                targets,
                to_head_queries,
                to_head_query_pos,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
        else:
            ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
                enc_memory,
                to_head_det_outputs,
                targets,
                to_head_queries,
                to_head_query_pos,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
        head_outputs = {}
        len_pfeats = len(pfeats_out)
        if self.training:
            head_outputs["losses"] = {"loss_l2p": -self.l2p_loss_weight * sim_pull}
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
            n_losses = len(self.loss_layers)
            if n_losses > 1:
                head_aux_outputs = []
                lvl_assign_ids, lvl_as_s, lvl_pfeats, lvl_logits = [], [], [], []
                loss_i_offset = len_pfeats - n_losses
                for i in range(n_losses):
                    (
                        bn_ids,
                        bn_asc,
                        bn_ptfeats,
                        bn_logits,
                    ) = self.split_id_asc_ptfeat_logits(
                        assign_ids,
                        as_s,
                        to_head_det_outputs,
                        pfeats_out[i + loss_i_offset],
                    )
                    lvl_assign_ids.append(bn_ids)
                    lvl_as_s.append(bn_asc)
                    lvl_pfeats.append(bn_ptfeats)
                    lvl_logits.append(bn_logits)
                    if i < n_losses - 1:
                        head_aux_outputs.append(
                            {"assign_ids": assign_ids[:, :n_queries]}
                        )
                    else:
                        head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                aux_losses = {}
                for i in range(n_losses):
                    li_pfeats_mtc = torch.cat(lvl_pfeats[i], dim=0)
                    li_pfeats_cls = self.bottelnecks[i](li_pfeats_mtc)
                    li_assign_ids = torch.cat(lvl_assign_ids[i])
                    li_as_s = (
                        torch.cat(lvl_as_s[i]) if lvl_as_s[i][0] is not None else None
                    )
                    li_logits = torch.cat(lvl_logits[i])
                    losses = self.compute_losses_lvl(
                        li_pfeats_cls,
                        li_assign_ids,
                        li_as_s,
                        li_logits,
                        loss_layer_lvl=i,
                    )
                    if self.metric_loss is not None:
                        # TODO choose neck feat
                        metric_losses = self.compute_metric_losses(
                            li_pfeats_mtc,
                            li_assign_ids,
                            li_as_s,
                            li_logits,
                        )
                        losses.update(metric_losses)
                    if i < n_losses - 1:
                        for k, v in losses.items():
                            aux_losses[k + "_{}".format(i)] = v
                    else:
                        head_outputs["losses"].update(losses)
                head_outputs["losses"].update(aux_losses)
                head_outputs["aux_outputs"] = head_aux_outputs

            else:
                (
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    assign_ids,
                    as_s,
                    to_head_det_outputs,
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bottelnecks[-1](pfeats_mtc)
                valid_ids = torch.cat(bn_ids)
                valid_as_s = torch.cat(bn_asc) if bn_asc[0] is not None else None
                valid_logits = torch.cat(bn_logits)
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    valid_ids,
                    valid_as_s,
                    valid_logits,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_mtc,
                        valid_ids,
                        valid_as_s,
                        valid_logits,
                    )
                    losses.update(metric_losses)
                head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(losses)
            if get_event_storage().iter % self.vis_period == 0:
                self.d_points_vis(
                    det_outputs["pred_logits"].squeeze(2).sigmoid(),
                    ref_pts[:, :n_queries],
                    [spts[:, :n_queries] for spts in samp_pts],
                    [satts[:, :n_queries] for satts in samp_atts],
                    targets,
                )
        else:
            pfeats_cls = self.bottelnecks[-1](pfeats_out[-1])
            head_outputs["reid_feats"] = tF.normalize(pfeats_cls, dim=-1)
        return head_outputs


class PstrDecoderHead(DTransDecoderHead):
    def __init__(self, head_cfg):
        super().__init__(head_cfg)
        self.query_gradient_idx = head_cfg.POS_BP_IDXS
        self.loss_layers = _get_clones(
            self.loss_layers[0], 3
        )  # compatible with compute loss method
        self.bottelnecks = _get_clones(self.bottelnecks, 3)

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]

        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
        if self.training:
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
        head_outputs = {}
        if not self.training:
            head_outputs["reid_feats"] = []
        else:
            head_outputs["losses"] = {}
        for mi, mem in enumerate(enc_memory):
            ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
                mem,
                to_head_det_outputs,
                targets,
                to_head_queries
                if mi == self.query_gradient_idx
                else to_head_queries.detach(),
                None,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
            if self.training:
                pfeats_cls = self.bottelnecks[mi][-1](pfeats_out[-1])
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    assign_ids,
                    as_s,
                    to_head_det_outputs["pred_logits"],
                    loss_layer_lvl=mi,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_out[-1],
                        assign_ids,
                        as_s,
                        to_head_det_outputs["pred_logits"],
                    )
                    losses.update(metric_losses)
                if mi == self.query_gradient_idx:
                    head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(
                    {k + str(mi): v for k, v in losses.items()}
                )
                if (
                    get_event_storage().iter % self.vis_period == 0
                    and mi == self.query_gradient_idx
                ):
                    self.d_points_vis(
                        det_outputs["pred_logits"].squeeze(2).sigmoid(),
                        ref_pts[:, :n_queries],
                        [spts[:, :n_queries] for spts in samp_pts],
                        [satts[:, :n_queries] for satts in samp_atts],
                        targets,
                    )
            else:
                pfeats_cls = self.bottelnecks[mi][-1](pfeats_out[-1])
                head_outputs["reid_feats"].append(pfeats_cls)
        if "reid_feats" in head_outputs:
            head_outputs["reid_feats"] = tF.normalize(
                torch.cat(head_outputs["reid_feats"], dim=-1), dim=-1
            )
        return head_outputs


class PstrDecoderHeadClean(PstrDecoderHead, DTransDecoderHeadClean):
    def __init__(self, head_cfg):
        PstrDecoderHead.__init__(head_cfg)

    def forward(
        self,
        enc_memory,
        det_outputs,
        targets,
        det_match_indices,
        queries,
        query_pos,
        pts_valid_ratios,
        memory_spatial_shapes,
        memory_level_start_index,
        memory_mask_flatten,
        img_aug_whwh=None,
    ):
        # det_outputs, targets, xyxy_abs boxes
        n_queries = queries.shape[1]
        if self.multi_layer_query and self.training:
            to_head_det_outputs = {}
            det_out_keys = [k for k in list(det_outputs.keys()) if "aux" not in k]
            for k in det_out_keys:
                vals = [det_outputs[k]] + [vd[k] for vd in det_outputs["aux_outputs"]]
                to_head_det_outputs[k] = torch.cat(vals, dim=1)
            to_head_queries = to_head_det_outputs["pred_embs"]

        else:
            to_head_det_outputs = det_outputs
            to_head_queries = queries
        if self.training:
            if self.multi_layer_query:
                assign_ids, as_s = [], []
                assign_ids_p, as_s_p = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
                assign_ids.append(assign_ids_p)
                as_s.append(as_s_p)
                for ai, aux_det in enumerate(det_outputs["aux_outputs"]):
                    assign_ids_a, as_s_a = self._id_assign_method(
                        aux_det, targets, det_match_indices[ai]
                    )
                    assign_ids.append(assign_ids_a)
                    as_s.append(as_s_a)
                assign_ids = torch.cat(assign_ids, dim=1)
                as_s = None if as_s[0] is None else torch.cat(as_s, dim=1)
            else:
                assign_ids, as_s = self._id_assign_method(
                    det_outputs, targets, det_match_indices[-1]
                )
        head_outputs = {}
        if not self.training:
            head_outputs["reid_feats"] = []
        else:
            head_outputs["losses"] = {}
        for mi, mem in enc_memory:
            ref_pts, samp_pts, samp_atts, pfeats_out = self.get_pfeats(
                mem,
                to_head_det_outputs,
                targets,
                to_head_queries
                if mi == self.query_gradient_idx
                else to_head_queries.detach(),
                None,
                pts_valid_ratios,
                memory_spatial_shapes,
                memory_level_start_index,
                memory_mask_flatten,
                img_aug_whwh,
            )
            if self.training:
                (
                    bn_ids,
                    bn_asc,
                    bn_ptfeats,
                    bn_logits,
                ) = self.split_id_asc_ptfeat_logits(
                    assign_ids,
                    as_s,
                    to_head_det_outputs,
                    pfeats_out[-1],
                )
                pfeats_mtc = torch.cat(bn_ptfeats, dim=0)
                pfeats_cls = self.bottelnecks[mi][-1](pfeats_mtc)
                valid_ids = torch.cat(bn_ids)
                valid_as_s = torch.cat(bn_asc) if bn_asc[0] is not None else None
                valid_logits = torch.cat(bn_logits)
                losses = self.compute_losses_lvl(
                    pfeats_cls,
                    valid_ids,
                    valid_as_s,
                    valid_logits,
                    loss_layer_lvl=mi,
                )
                if self.metric_loss is not None:
                    metric_losses = self.compute_metric_losses(
                        pfeats_mtc,
                        valid_ids,
                        valid_as_s,
                        valid_logits,
                    )
                    losses.update(metric_losses)
                if mi == self.query_gradient_idx:
                    head_outputs["assign_ids"] = assign_ids[:, :n_queries]
                # head_outputs["reid_feats"] = reid_feats
                head_outputs["losses"].update(
                    {k + str(mi): v for k, v in losses.items()}
                )
                if (
                    get_event_storage().iter % self.vis_period == 0
                    and mi == self.query_gradient_idx
                ):
                    self.d_points_vis(
                        det_outputs["pred_logits"].squeeze(2).sigmoid(),
                        ref_pts[:, :n_queries],
                        [spts[:, :n_queries] for spts in samp_pts],
                        [satts[:, :n_queries] for satts in samp_atts],
                        targets,
                    )
            else:
                pfeats_cls = self.bottelnecks[mi](pfeats_out[-1])
                head_outputs["reid_feats"].append(pfeats_cls, dim=-1)
        if "reid_feats" in head_outputs:
            head_outputs["reid_feats"] = tF.normalize(
                torch.cat(head_outputs["reid_feats"], dim=-1), dim=-1
            )
        return head_outputs


class ContrastDecoderHead(DTransDecoderHeadClean):
    def __init__(self, head_cfg):
        super(ReidHeadBase, self).__init__()
        self.head_cfg = head_cfg
        if hasattr(head_cfg.ID_ASSIGN, "POS_IOU_THRED"):
            self.pos_iou_thred = head_cfg.ID_ASSIGN.POS_IOU_THRED
        if hasattr(head_cfg.ID_ASSIGN, "POS_CENTER_RATIO"):
            self.pos_center_ratio = head_cfg.ID_ASSIGN.POS_CENTER_RATIO
        self.pos_score_thred = head_cfg.ID_ASSIGN.POS_SCORE_THRED
        self._id_assign_method = self._get_id_assign_method(head_cfg.ID_ASSIGN.NAME)
        self.pfeat_dim = head_cfg.PERSON_FEATURE.DIM
        self.box_gradient = head_cfg.BOX_BP
        self.metric_loss = None
        # visualization
        self.vis_period = head_cfg.VIS_PERIOD
        self.vis_inf = head_cfg.VIS_INF
        self.vis_inf_dir = head_cfg.VIS_INF_SAVE
        if self.vis_inf and not os.path.exists(self.vis_inf_dir):
            os.makedirs(self.vis_inf_dir)
        # contrast
        self.c_temp = head_cfg.LOSS.CONTRAST.TEMP
        self.contrast_lw = head_cfg.LOSS.LOSS_WEIGHTS.CONTRAST

        # create the queue
        len_queue = head_cfg.LOSS.CONTRAST.CQ_SIZE
        self.register_buffer("queue", torch.randn(self.pfeat_dim, len_queue))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # decoders
        dec_sa = head_cfg.SA
        self.dec_norm = head_cfg.NORM
        dec_ca_res = head_cfg.CA_RES
        ca_mode = head_cfg.CA_MODE
        dec_layer = DeformableTransformerDecoderLayerSA(
            d_model=head_cfg.PERSON_FEATURE.DIM,
            d_ffn=head_cfg.DIM_FEEDFORWARD,
            dropout=head_cfg.DROPOUT,
            n_levels=head_cfg.N_FEATURE_LEVELS,
            n_heads=head_cfg.N_HEADS,
            n_points=head_cfg.N_POINTS,
            sa_mode=dec_sa,
            norm=self.dec_norm,
            ca_res=dec_ca_res,
            ca_mode=ca_mode,
        )
        self.dec_layers = _get_clones(dec_layer, head_cfg.N_LAYERS)
        self.query_gradient = head_cfg.QUERY_BP
        self.pos_gradient = head_cfg.POS_BP
        self.vis_period = head_cfg.VIS_PERIOD
        self.bottelnecks = nn.ModuleList()
        bt_neck_type = head_cfg.BOTTELNECK
        self.out_lvls = head_cfg.LOSS.AUX_LOSS + 1
        for li in range(self.out_lvls):
            self.bottelnecks.append(get_bt_neck(bt_neck_type, self.pfeat_dim))
        self.multi_layer_query = head_cfg.MULTI_LAYER_QUERY

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        comm.synchronize()
        all_keys = comm.all_gather(keys)
        all_keys = [k.to(self.queue.device) for k in all_keys]
        keys = torch.cat(all_keys, dim=0)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        target_ptr = ptr + batch_size
        if target_ptr > self.queue.shape[1]:
            left_len = self.queue.shape[1] - ptr
            self.queue[:, ptr : ptr + left_len] = keys[:left_len].T
            self.queue[:, : batch_size - left_len] = keys[left_len:].T
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr : ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.queue.shape[1]  # move pointer

        self.queue_ptr[0] = ptr

    def compute_losses_lvl(
        self, pfeats, assigned_ids, assign_scores, det_logits, loss_layer_lvl=-1
    ):
        pfeats_norm = tF.normalize(pfeats, dim=1)
        l_neg_all = (
            torch.einsum("nc,ck->nk", [pfeats_norm, self.queue.clone().detach()])
            / self.c_temp
        )
        # iter by id
        uniqe_pids = torch.unique(assigned_ids)
        p_losses = []
        for pid in uniqe_pids:
            pid_mask = assigned_ids == pid
            # NOTE ignore single for now
            if pid_mask.sum() < 2:
                continue
            if pid == -2:  # bg
                continue
            pid_embeddings = pfeats_norm[pid_mask]
            l_pos = (
                torch.einsum("nc,ck->nk", [pid_embeddings, pid_embeddings.T])
                / self.c_temp
            )  # self similarity is included, np x np
            l_neg = l_neg_all[pid_mask]  # np x ng
            np, ng = l_pos.shape[0], l_neg.shape[1]
            select_cvt = torch.eye(np, dtype=torch.bool, device=self.queue.device)
            select_mask = torch.logical_not(select_cvt)
            l_pos_expd = l_pos[select_mask].unsqueeze(1)  # (np x np-1) x 1

            # l_pos_expd = l_pos.view(-1, 1)  # (np x np) x 1

            l_neg_expd = l_neg.unsqueeze(1).expand(-1, np - 1, -1)
            l_neg_expd = l_neg_expd.reshape(-1, ng)
            l_all = torch.cat([l_pos_expd, l_neg_expd], dim=-1)
            labels = torch.zeros(
                l_all.shape[0], dtype=torch.long, device=self.queue.device
            )  # (np x np-1)
            id_losses = tF.cross_entropy(l_all, labels, reduction="none")  # (np x np-1)
            id_losses = id_losses.view(np, np - 1)
            id_losses = id_losses.mean(dim=1)
            if torch.isnan(id_losses).sum() > 0:
                print("get")
            p_losses.append(id_losses)
        if len(p_losses) > 0:
            p_losses = torch.cat(p_losses, dim=0)
            n_inst = torch.as_tensor(
                [p_losses.shape[0]], dtype=torch.float, device=self.queue.device
            )
        else:
            p_losses = torch.tensor(
                0, dtype=pfeats_norm.dtype, device=pfeats_norm.device
            )
            n_inst = torch.as_tensor([0], dtype=torch.float, device=self.queue.device)
        comm.synchronize()
        all_n_inst = comm.all_gather(n_inst)
        num_inst = sum([num.to("cpu") for num in all_n_inst])
        num_inst = torch.clamp(num_inst / comm.get_world_size(), min=1).item()
        reid_loss = p_losses.sum() / num_inst
        if torch.isnan(reid_loss):
            print("get")
        self._dequeue_and_enqueue(pfeats_norm)
        return {"loss_contrast": reid_loss * self.contrast_lw}
