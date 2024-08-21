#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from psd2.modeling.poolers import ROIPooler
from psd2.structures import Boxes
from detectron2.modeling.anchor_generator import build_anchor_generator
from .extra_deconv import CenternetDeconv

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):
    def __init__(
        self,
        pooler_type,
        in_feats,
        pooler_size,
        pooler_samp_ratio,
        roi_input_shape,
        rcnn_head,
        num_heads,
        deep_sup,
        num_cls,
        use_focal=True,
        focal_prior_prob=0.01,
        pfeat_type="null",
    ):
        super().__init__()
        self.pfeat_type = pfeat_type
        # Build RoI.
        box_pooler = self._init_box_pooler(
            pooler_type, in_feats, pooler_size, pooler_samp_ratio, roi_input_shape
        )
        self.box_pooler = box_pooler

        # Build heads.

        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = deep_sup

        # Init parameters.
        self.use_focal = use_focal
        self.num_classes = num_cls
        if self.use_focal:
            prior_prob = focal_prior_prob
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(
        pooler_type, in_features, pooler_resolution, sampling_ratio, input_shape
    ):

        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()
        if self.pfeat_type == "null":
            for rcnn_head in self.head_series:

                class_logits, pred_bboxes, proposal_features = rcnn_head(
                    features, bboxes, proposal_features, self.box_pooler
                )

                if self.return_intermediate:
                    inter_class_logits.append(class_logits)
                    inter_pred_bboxes.append(pred_bboxes)
                bboxes = pred_bboxes.detach()

            if self.return_intermediate:
                return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

            return class_logits[None], pred_bboxes[None]
        else:
            inter_pfeats = []
            for rcnn_head in self.head_series:

                class_logits, pred_bboxes, proposal_features, pfeats = rcnn_head(
                    features, bboxes, proposal_features, self.box_pooler
                )

                if self.return_intermediate:
                    inter_class_logits.append(class_logits)
                    inter_pred_bboxes.append(pred_bboxes)
                    inter_pfeats.append(pfeats)
                bboxes = pred_bboxes.detach()

            if self.return_intermediate:
                return (
                    torch.stack(inter_class_logits),
                    torch.stack(inter_pred_bboxes),
                    torch.stack(inter_pfeats),
                )

            return class_logits[None], pred_bboxes[None], pfeats[None]


class RCNNHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_cls,
        num_cls_layers,
        num_reg_layers,
        use_focal,
        inst_interact,
        norm="ln",
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        pfeat_type="null",
    ):
        super().__init__()

        self.pfeat_type = pfeat_type

        self.d_model = hidden_dim
        # dynamic.
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.inst_interact = inst_interact

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        norm_type = {"ln": nn.LayerNorm, "bn": nn.BatchNorm1d}[norm]
        self.norm1 = norm_type(hidden_dim)
        self.norm2 = norm_type(hidden_dim)
        self.norm3 = norm_type(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        cls_module = list()
        for _ in range(num_cls_layers):
            cls_module.append(nn.Linear(hidden_dim, hidden_dim, False))
            cls_module.append(norm_type(hidden_dim))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_layers):
            reg_module.append(nn.Linear(hidden_dim, hidden_dim, False))
            reg_module.append(norm_type(hidden_dim))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = use_focal
        if self.use_focal:
            self.class_logits = nn.Linear(hidden_dim, num_cls)
        else:
            self.class_logits = nn.Linear(hidden_dim, num_cls + 1)
        self.bboxes_delta = nn.Linear(hidden_dim, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """
        return_pfeats = {}
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        _, n_ch, rh, rw = roi_features.shape
        return_pfeats["pre_iou"] = roi_features.view(N, nr_boxes, n_ch, rh, rw)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(
            2, 0, 1
        )

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[
            0
        ]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = (
            pro_features.view(nr_boxes, N, self.d_model)
            .permute(1, 0, 2)
            .reshape(1, N * nr_boxes, self.d_model)
        )
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features)))
        )
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        return_pfeats["sh_prop"] = fc_feature.clone().view(N, nr_boxes, -1)
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        if self.pfeat_type != "null":
            return (
                class_logits.view(N, nr_boxes, -1),
                pred_bboxes.view(N, nr_boxes, -1),
                obj_features,
                return_pfeats[self.pfeat_type],
            )
        return (
            class_logits.view(N, nr_boxes, -1),
            pred_bboxes.view(N, nr_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):
    def __init__(
        self, hidden_dim, dim_dynamic, num_dynamic, pooler_resolution, norm="ln"
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(
            self.hidden_dim, self.num_dynamic * self.num_params
        )
        norm_type = {"ln": nn.LayerNorm, "bn": nn.BatchNorm1d}[norm]
        self.norm1 = norm_type(self.dim_dynamic)
        self.norm2 = norm_type(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.hidden_dim * pooler_resolution[0] * pooler_resolution[1]
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = norm_type(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(
            -1, self.hidden_dim, self.dim_dynamic
        )
        param2 = parameters[:, :, self.num_params :].view(
            -1, self.dim_dynamic, self.hidden_dim
        )

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# NOTE FOR OnePS
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Build heads.
        num_classes = cfg.DETECTOR.MODEL.HEAD.NUM_CLASSES
        d_model = cfg.MODEL.FPN.OUT_CHANNELS
        activation = cfg.DETECTOR.MODEL.HEAD.ACTIVATION
        num_conv = cfg.DETECTOR.MODEL.HEAD.NUM_CONV
        conv_norm = cfg.DETECTOR.MODEL.HEAD.CONV_NORM
        num_levels = len(cfg.DETECTOR.MODEL.IN_FEATURES)
        conv_channels = cfg.DETECTOR.MODEL.HEAD.CONV_CHANNELS

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(num_levels)])
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)
        self.features_stride = cfg.DETECTOR.MODEL.HEAD.FEATURES_STRIDE

        cls_conv_module = list()
        for idx in range(num_conv):
            if idx == 0:
                cls_conv_module.append(
                    nn.Conv2d(
                        d_model,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            else:
                cls_conv_module.append(
                    nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            if conv_norm == "GN":
                cls_conv_module.append(nn.GroupNorm(32, conv_channels))
            else:
                cls_conv_module.append(nn.BatchNorm2d(conv_channels))
            cls_conv_module.append(nn.ReLU(inplace=True))

        self.cls_conv_module = nn.ModuleList(cls_conv_module)

        reg_conv_module = list()
        for idx in range(num_conv):
            if idx == 0:
                reg_conv_module.append(
                    nn.Conv2d(
                        d_model,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            else:
                reg_conv_module.append(
                    nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            if conv_norm == "GN":
                reg_conv_module.append(nn.GroupNorm(32, conv_channels))
            else:
                reg_conv_module.append(nn.BatchNorm2d(conv_channels))
            reg_conv_module.append(nn.ReLU(inplace=True))

        self.reg_conv_module = nn.ModuleList(reg_conv_module)

        self.cls_score = nn.Conv2d(
            conv_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.ltrb_pred = nn.Conv2d(conv_channels, 4, kernel_size=3, stride=1, padding=1)

        # Init parameters.
        prior_prob = cfg.DETECTOR.LOSS.FOCAL.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)

    def forward(self, features_list):
        class_logits = list()
        pred_bboxes = list()
        locationall = list()
        fpn_levels = list()

        for l, stride_feat in enumerate(features_list):
            cls_feat = stride_feat
            reg_feat = stride_feat

            for conv_layer in self.cls_conv_module:
                cls_feat = conv_layer(cls_feat)

            for conv_layer in self.reg_conv_module:
                reg_feat = conv_layer(reg_feat)

            locations = self.locations(stride_feat, self.features_stride[l])[None]

            stride_class_logits = self.cls_score(cls_feat)
            reg_ltrb = self.ltrb_pred(reg_feat)

            scale_reg_ltrb = self.scales[l](reg_ltrb)
            stride_pred_ltrb = F.relu(scale_reg_ltrb) * self.features_stride[l]
            stride_pred_bboxes = self.apply_ltrb(locations, stride_pred_ltrb)
            bs, c, h, w = stride_class_logits.shape
            bs, four, h, w = stride_pred_bboxes.shape
            class_logits.append(stride_class_logits.view(bs, c, -1))
            pred_bboxes.append(stride_pred_bboxes.view(bs, four, -1))

            locationall.append(locations.view(1, 2, -1).repeat(bs, 1, 1))
            fpn_levels.append(locations.new_ones(bs, 1, h * w) * l)

        class_logits = torch.cat(class_logits, dim=-1)
        pred_bboxes = torch.cat(pred_bboxes, dim=-1)
        locationall = torch.cat(locationall, dim=-1)
        fpn_levels = torch.cat(fpn_levels, dim=-1)

        return features_list, class_logits, pred_bboxes, None, locationall, fpn_levels

    def apply_ltrb(self, locations, pred_ltrb):
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes

    @torch.no_grad()
    def locations(self, features, stride):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = torch.arange(
            0, w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations


class RetinaHead(nn.Module):
    def __init__(
        self,
        cfg,
        feature_shapes,
        weights=[1.0, 1.0, 1.0, 1.0],
        scale_clamp=_DEFAULT_SCALE_CLAMP,
    ):
        super().__init__()
        self.weights = weights
        self.scale_clamp = scale_clamp

        # Build heads.
        num_classes = cfg.DETECTOR.MODEL.HEAD.NUM_CLASSES
        d_model = cfg.MODEL.FPN.OUT_CHANNELS
        activation = cfg.DETECTOR.MODEL.HEAD.ACTIVATION
        num_conv = cfg.DETECTOR.MODEL.HEAD.NUM_CONV
        conv_norm = cfg.DETECTOR.MODEL.HEAD.CONV_NORM
        num_levels = len(cfg.DETECTOR.MODEL.IN_FEATURES)
        conv_channels = cfg.DETECTOR.MODEL.HEAD.CONV_CHANNELS

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)
        self.features_stride = cfg.DETECTOR.MODEL.HEAD.FEATURES_STRIDE

        cls_conv_module = list()
        for idx in range(num_conv):
            if idx == 0:
                cls_conv_module.append(
                    nn.Conv2d(
                        d_model,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            else:
                cls_conv_module.append(
                    nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )

            cls_conv_module.append(nn.ReLU(inplace=True))

        self.cls_conv_module = nn.ModuleList(cls_conv_module)

        reg_conv_module = list()
        for idx in range(num_conv):
            if idx == 0:
                reg_conv_module.append(
                    nn.Conv2d(
                        d_model,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
            else:
                reg_conv_module.append(
                    nn.Conv2d(
                        conv_channels,
                        conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )

            reg_conv_module.append(nn.ReLU(inplace=True))

        self.reg_conv_module = nn.ModuleList(reg_conv_module)

        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.anchor_generator = anchor_generator
        num_anchors = anchor_generator.num_cell_anchors
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        self.num_anchors = num_anchors[0]

        self.cls_score = nn.Conv2d(
            conv_channels,
            self.num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bbox_pred = nn.Conv2d(
            conv_channels, self.num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # Init parameters.
        prior_prob = cfg.DETECTOR.LOSS.FOCAL.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)

    def forward(self, features_list):
        class_logits = list()
        pred_bboxes = list()
        anchorall = list()
        anchors = self.anchor_generator(features_list)

        for l, (anchor, stride_feat) in enumerate(zip(anchors, features_list)):
            cls_feat = stride_feat
            reg_feat = stride_feat

            for conv_layer in self.cls_conv_module:
                cls_feat = conv_layer(cls_feat)

            for conv_layer in self.reg_conv_module:
                reg_feat = conv_layer(reg_feat)

            stride_class_logits = self.cls_score(cls_feat)
            stride_bbox_pred = self.bbox_pred(reg_feat)

            stride_pred_bboxes = self.apply_deltas(
                stride_bbox_pred, anchor.tensor[None]
            )
            bs, ac, h, w = stride_class_logits.shape
            stride_class_logits = stride_class_logits.reshape(
                bs, -1, self.num_classes, h, w
            )
            stride_class_logits = stride_class_logits.permute(0, 2, 3, 4, 1)

            class_logits.append(stride_class_logits.reshape(bs, self.num_classes, -1))
            pred_bboxes.append(stride_pred_bboxes.view(bs, 4, -1))

            anchorall.append(anchor.tensor[None].permute(0, 2, 1).repeat(bs, 1, 1))

        class_logits = torch.cat(class_logits, dim=-1)
        pred_bboxes = torch.cat(pred_bboxes, dim=-1)
        anchorall = torch.cat(anchorall, dim=-1)

        return features_list, class_logits, pred_bboxes, anchorall, None, None

    def apply_deltas(self, pred_deltas, anchor):
        """
        pred_deltas:  (N, A4, H, W)
        anchor:  (1, HWA, 4)

        return:
        pred_bboxes: (N, 4, HWA)
        """

        bs, afour, h, w = pred_deltas.shape
        deltas = pred_deltas.permute(0, 2, 3, 1).reshape(bs, -1, 4)  # (N, HWA, 4)

        boxes = anchor.to(deltas.dtype)  # (1, HWA, 4)

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dw = deltas[:, :, 2] / ww
        dh = deltas[:, :, 3] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes.permute(0, 2, 1)  # (N, 4, HWA)


class CenterNetHead(nn.Module):
    def __init__(self, cfg, backbone_shape=[2048, 1024, 512, 256]):
        super().__init__()

        # Build heads.
        num_classes = cfg.DETECTOR.MODEL.HEAD.NUM_CLASSES
        d_model = cfg.DETECTOR.MODEL.HEAD.DECONV_CHANNEL[-1]
        activation = cfg.DETECTOR.MODEL.HEAD.ACTIVATION

        self.deconv = CenternetDeconv(
            channels=cfg.DETECTOR.MODEL.HEAD.DECONV_CHANNEL,
            deconv_kernel=cfg.DETECTOR.MODEL.HEAD.DECONV_KERNEL,
            modulate_deform=cfg.DETECTOR.MODEL.HEAD.MODULATE_DEFORM,
            in_features=cfg.DETECTOR.MODEL.IN_FEATURES,
            with_dcn=cfg.DETECTOR.MODEL.HEAD.DCN,
            backbone_shape=backbone_shape,
        )

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = _get_activation_fn(activation)

        self.feat1 = nn.Conv2d(
            self.d_model, self.d_model, kernel_size=3, stride=1, padding=1
        )
        self.cls_score = nn.Conv2d(
            d_model, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.ltrb_pred = nn.Conv2d(d_model, 4, kernel_size=3, stride=1, padding=1)

        # Init parameters.
        prior_prob = cfg.DETECTOR.LOSS.FOCAL.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # initialize the bias for focal loss.
        nn.init.constant_(self.cls_score.bias, self.bias_value)

    def forward(self, features_list):

        features = self.deconv(features_list)
        locations = self.locations(features)[None]

        feat = self.activation(self.feat1(features))

        class_logits = self.cls_score(feat)
        pred_ltrb = F.relu(self.ltrb_pred(feat))
        pred_bboxes = self.apply_ltrb(locations, pred_ltrb)

        bs = pred_bboxes.shape[0]
        locationall = locations.repeat(bs, 1, 1, 1)

        return [feat], class_logits, pred_bboxes, None, locationall, None

    def apply_ltrb(self, locations, pred_ltrb):
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes

    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = torch.arange(
            0, w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations
