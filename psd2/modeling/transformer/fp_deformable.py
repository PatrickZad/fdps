from torch import nn
from .deformable import (
    _get_clones,
    DeformableTransformerEncoderLayer,
    DeformableTransformerEncoder,
    _get_activation_fn,
    _inverse_sigmoid,
)
import warnings
from psd2.layers.ops.modules import MSDeformAttn
from psd2.layers.ops.modules.ms_deform_attn import _is_power_of_2
from psd2.layers.ops.functions import MSDeformAttnFunction
import torch
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import math
import torch.nn.functional as F


class FP_MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_levels=4,
        n_heads=8,
        n_points=4,
        mask_obj_query=False,
        r_samp_pt=False,
    ):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )
        self.mask_object_query = mask_obj_query
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        self.r_samp_pt = r_samp_pt

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
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
        sample_attention_area: N x Length_{query} x nhead(1) x nlevel x npoint(1) x 4, xyxy box in padding rel for box masked dformable attention
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
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        # add support for attn mask
        if self.mask_object_query:
            attn_mask = torch.zeros(
                (Len_q, self.n_heads, self.n_levels, self.n_points),
                dtype=torch.bool,
                device=attention_weights.device,
            )  # broadcasted over bs
            attn_mask[:, :, -1, :] = True  # the last feature level is object queries
            assert (
                attn_mask.dtype == torch.bool
            ), "only support torch.bool as attn_mask for now, got {}".format(
                attn_mask.dtype
            )
            attention_weights.masked_fill_(attn_mask, float("-inf"))
        attention_weights = F.softmax(
            attention_weights.view(
                N, Len_q, self.n_heads, self.n_levels * self.n_points
            ),
            -1,
        ).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

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


class FPDeformableEncTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_enchead_layers=6,
        query_shape=(15, 20),
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_head=False,
        num_feature_levels=4,
        enc_n_points=4,
        mask_obj_queries=False,
        pattn_share=False,
    ):
        super().__init__()
        self.query_shape = query_shape
        self.num_queries = query_shape[0] * query_shape[1]
        self.d_model = d_model
        self.nhead = nhead
        enc_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(enc_layer, num_encoder_layers)
        # enc det head
        enchead_layer = FPDeformableHeadEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels + 1,
            nhead,
            enc_n_points,
            mask_obj_queries,
        )
        self.enc_head = FPDeformableEncHead(
            enchead_layer,
            num_enchead_layers,
            d_model,
            nhead,
            dropout,
            self.num_queries,
            return_intermediate_head,
            pattn_share,
        )
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn) or isinstance(m, FP_MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed):
        assert query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # (B, H*W, embed_dims)
        mask_flatten_img = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(
            lvl_pos_embed_flatten, 1
        )  # (B, H*W, embed_dims)
        spatial_shapes_img = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index_img = torch.cat(
            (
                spatial_shapes_img.new_zeros((1,)),
                spatial_shapes_img.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios_img = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # add additional mask_query at the end, note False is not masked
        mask_query = torch.zeros(
            (bs, self.num_queries), dtype=mask.dtype, device=mask.device
        )
        mask_flatten.append(mask_query)
        mask_flatten = torch.cat(mask_flatten, 1)

        # add additional spatial_shape at the end
        spatial_shapes.append(self.query_shape)  # fake shape
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1
        )  # (B, levels, 2)
        # add additional valid_ratio at the end
        valid_ratios_query = torch.ones(
            (bs, 1, 2), dtype=valid_ratios.dtype, device=valid_ratios.device
        )
        valid_ratios = torch.cat(
            [valid_ratios, valid_ratios_query], dim=1
        )  # (B, levels + 1, 2)
        # encoder1 only handles the image tokens
        enc_output = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes_img,
            level_start_index=level_start_index_img,
            valid_ratios=valid_ratios_img,
            pos=lvl_pos_embed_flatten,
            padding_mask=mask_flatten_img,
        )
        # get queries here
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # reference points for query tokens
        reference_points_query = self.reference_points(query_pos).sigmoid()
        init_reference_query_out = reference_points_query  # (B, 300, 2)
        # reference points on object queries automatically added,  (B, Len_seq, levels+1, 2)
        reference_points_query = (
            reference_points_query[:, :, None] * valid_ratios[:, None]
        )

        # query = query.permute(1, 0, 2)
        # query_pos = query_pos.permute(1, 0, 2)
        # final reference_points (B, Len_seq, levels+1, 2)
        reference_points_imgtoken = DeformableTransformerEncoder.get_reference_points(
            spatial_shapes, valid_ratios, spatial_shapes_img.device
        )[:, : -self.num_queries, ...]
        query_feat_reference_points = torch.cat(
            [reference_points_imgtoken, reference_points_query], dim=1
        )

        query_feat = torch.cat([enc_output, query], dim=1)
        query_feat_pos = torch.cat([lvl_pos_embed_flatten, query_pos], dim=1)
        hs, inter_references = self.enc_head(
            query_feat_reference_points,
            query_feat,
            spatial_shapes,
            level_start_index,
            query_feat_pos,
            mask_flatten,
        )
        return (
            (
                spatial_shapes_img,
                level_start_index[:-1],
                valid_ratios_img,
                mask_flatten_img,
            ),
            hs,
            init_reference_query_out,
            inter_references,
            enc_output,
        )


class FPDeformableEncHead(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model,
        sattn_n_heads,
        sattn_dropout,
        n_queries,
        return_intermediate=False,
        pattn_share=False,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.num_queries = n_queries
        self.pattn = nn.MultiheadAttention(
            d_model, sattn_n_heads, dropout=sattn_dropout
        )
        if pattn_share:
            for enc_layer in self.layers:
                enc_layer.prompt_attn = self.pattn
        else:
            self.pattn = _get_clones(self.pattn, self.num_layers)
            for enc_layer, li_pattn in zip(self.layers, self.pattn):
                enc_layer.prompt_attn = li_pattn

    def forward(
        self,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        query_pos=None,
        src_padding_mask=None,
    ):
        intermediate = []
        intermediate_reference_points = []
        output = src
        for lid, layer in enumerate(self.layers):
            # ref pts multipled ratios outside
            output = layer(
                output,
                query_pos,
                reference_points,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + _inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + _inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output[:, -self.num_queries :, ...])
                intermediate_reference_points.append(
                    reference_points[:, -self.num_queries :, -1, ...]
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return (
            output[:, -self.num_queries, ...],
            reference_points[:, -self.num_queries :, -1, ...],
        )


class FPDeformableHeadEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        mask_obj_quries=False,
    ):
        super().__init__()
        # prompt attention
        self.prompt_attn = None  # hack impl
        self.normp = nn.LayerNorm(d_model)
        self.dropoutp = nn.Dropout(dropout)
        # self attention
        self.self_attn = FP_MSDeformAttn(
            d_model, n_levels, n_heads, n_points, mask_obj_quries
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # prompt attention
        object_queies_start_idx = level_start_index[-1]
        image_tokens = src[:, :object_queies_start_idx, ...]
        object_queies = src[:, object_queies_start_idx:, ...]
        query_pos_propmt = pos[:, object_queies_start_idx:, ...]
        q = k = self.with_pos_embed(object_queies, query_pos_propmt)
        prompt_quries = self.prompt_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            object_queies.transpose(0, 1),
        )[0].transpose(0, 1)
        prompt_quries = object_queies + self.dropoutp(prompt_quries)
        prompt_quries = self.normp(prompt_quries)
        src = torch.cat([image_tokens, prompt_quries], dim=1)
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src
