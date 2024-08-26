from .ti_rcnn_baseline import *
import torch.utils.checkpoint as checkpoint


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side4(TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side, self).__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res2 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        self.side_res2 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 64,
                    "out_channels": 256,
                    "norm": "BN",
                    "bottleneck_channels": 64,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side, self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}, {}, {}]
            bn_neck_params = {}
            for si in range(4):
                res_name = "res{}".format(si + 2)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res2.load_state_dict(side_res_params[0])
            self.side_res3.load_state_dict(side_res_params[1])
            self.side_res4.load_state_dict(side_res_params[2])
            self.side_res5.load_state_dict(side_res_params[3])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res2_feat = checkpoint.checkpoint(
            self.side_res2, det_backbone_features["stem"]
        )
        alpha = torch.sigmoid(self.alpha_res2)
        reid_res2_feat = (
            alpha * det_backbone_features["res2"] + (1 - alpha) * reid_res2_feat
        )
        reid_res3_feat = checkpoint.checkpoint(self.side_res3, reid_res2_feat)
        del reid_res2_feat
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = checkpoint.checkpoint(self.side_res4, reid_res3_feat)
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = checkpoint.checkpoint(
            self.side_res5, p_feat_maps
        )  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side2(TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side, self).__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side, self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = [{}, {}]
            bn_neck_params = {}
            for si in range(2):
                res_name = "res{}".format(si + 4)
                for pn, pa in state_dict.items():
                    if res_name in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        side_res_params[si][pn.split(res_name)[1][1:]] = pa
                    elif "bn_neck" in pn:
                        if not isinstance(pa, torch.Tensor):
                            pa = torch.from_numpy(pa).to(self.device)
                        bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res4.load_state_dict(side_res_params[0])
            self.side_res5.load_state_dict(side_res_params[1])
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res4_feat = self.side_res4(det_backbone_features["res3"])
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side1(TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side, self).__init__(**kwargs)

        self.side_init = side_init
        # side reid network
        self.bn_neck = bn_neck
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side, self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = {}
            bn_neck_params = {}
            res_name = "res5"
            for pn, pa in state_dict.items():
                if res_name in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    side_res_params[pn.split(res_name)[1][1:]] = pa
                elif "bn_neck" in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res5.load_state_dict(side_res_params)
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Side0(TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side, self).__init__(**kwargs)

        self.side_init = side_init
        # side reid network
        self.bn_neck = bn_neck
        # freeze det params
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.proposal_generator.parameters():
            p.requires_grad_(False)
        for p in self.roi_heads.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4Side, self).load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            bn_neck_params = {}
            for pn, pa in state_dict.items():
                if "bn_neck" in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = self.roi_heads.res5(p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Joint(TiRCNN_C4SideJoint):
    @configurable
    def __init__(self, bn_neck, **kwargs) -> None:
        super(TiRCNN_C4SideJoint, self).__init__(**kwargs)

        # side reid network
        self.bn_neck = bn_neck

    def load_state_dict(self, *args, **kws):
        output = super(TiRCNN_C4SideJoint, self).load_state_dict(*args, **kws)
        return output

    @classmethod
    def from_config(cls, cfg):
        res = super(TiRCNN_C4SideJoint, cls).from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        return res

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = self.roi_heads.res5(p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Res5Joint(TiRCNN_Baseline):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super().__init__(**kwargs)

        self.side_init = side_init
        # side reid network
        self.bn_neck = bn_neck
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        res["side_init"] = cfg.REID_HEAD.INIT_WEIGHT
        return res

    def load_state_dict(self, *args, **kws):
        output = super().load_state_dict(*args, **kws)
        model_resume = True
        for param_k in output.missing_keys:
            if "side_res" in param_k:
                model_resume = False
                break
        if not model_resume and self.side_init != "":
            # model is not resuming from saved checkpoint
            # init side networks
            state_dict = _load_file(self.side_init)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            side_res_params = {}
            bn_neck_params = {}
            res_name = "res5"
            for pn, pa in state_dict.items():
                if res_name in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    side_res_params[pn.split(res_name)[1][1:]] = pa
                elif "bn_neck" in pn:
                    if not isinstance(pa, torch.Tensor):
                        pa = torch.from_numpy(pa).to(self.device)
                    bn_neck_params[pn.split("bn_neck")[1][1:]] = pa
            self.side_res5.load_state_dict(side_res_params)
            if len(bn_neck_params) > 0:
                self.bn_neck.load_state_dict(bn_neck_params)
        return output

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        return det_backbone_features["res4"]

    def get_reid_person_features(self, reid_backbone_features, person_boxes):
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in person_boxes]
        roi_feats = self.reid_pooler([reid_backbone_features], d2_boxes)
        return roi_feats

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = checkpoint.checkpoint(self.side_res5,p_feat_maps)  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            # forward det
            proposals, proposal_losses = self.proposal_generator(
                images, det_bk_features, gt_instances
            )

            # det_pred_instances, det_losses = self.roi_heads.forward_unms(
            #    images, det_bk_features, proposals, gt_instances
            # )
            det_pred_instances, det_losses = self.roi_heads(
                images, det_bk_features, proposals, gt_instances
            )
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    outputs = {"pred_boxes": [], "pred_scores": []}
                    for pi, pred_img in enumerate(det_pred_instances):
                        boxes = pred_img.pred_boxes.tensor
                        scores = pred_img.scores
                        outputs["pred_boxes"].append(boxes)
                        outputs["pred_scores"].append(scores.unsqueeze(1))
                    for gti in gt_instances:
                        gt_id = gti.gt_classes
                        gt_id[gt_id == 0] = -1
                        gt_id[gt_id > 0] -= 2
                    self.visualize_training(
                        (images, gt_instances),
                        det_bk_features[list(det_bk_features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(proposal_losses)
            # forward re-id
            det_pred = {
                "pred_boxes": [],
                "pred_scores": [],
                "pred_pids": [],
            }
            """
            for pred_img in det_pred_instances:
                det_pred["pred_boxes"].append(pred_img.pred_boxes.tensor)
                det_pred["pred_scores"].append(pred_img.scores)
            """
            # back to original pid
            for gti in gt_instances:
                cur_pids = gti.gt_classes
                gti.gt_classes[cur_pids == 0] = -1
                gti.gt_classes[cur_pids > 0] -= 2
            reid_bk_feats = self.get_reid_backbone_features(det_bk_features, images)
            pos_boxes, pos_ids, pos_scores = self.get_ps_pos_samples(
                det_pred, gt_instances
            )

            pos_featmaps = self.get_reid_person_features(
                reid_bk_feats, pos_boxes
            )  # nf x c x h x w

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:

                    self.visualize_training_ps(
                        images.tensor,
                        pos_featmaps.split([bxs.shape[0] for bxs in pos_boxes], dim=0),
                        pos_boxes,
                        pos_ids,
                    )
            del reid_bk_feats
            pos_embs = self.get_reid_embed(pos_featmaps)
            del pos_featmaps
            pos_ids = torch.cat(pos_ids, dim=0)
            rl = self.reid_loss(pos_embs, pos_ids, None)
            losses.update(rl)
            return losses
        else:
            return super().forward(input_list)


@META_ARCH_REGISTRY.register()
class TiRCNN_C4Coupled(TiRCNN_Baseline):
    @configurable
    def __init__(self, bn_neck, **kwargs) -> None:
        super().__init__(**kwargs)
        # side reid network
        self.bn_neck = bn_neck

    @classmethod
    def from_config(cls, cfg):
        res = super().from_config(cfg)
        bn_neck = nn.BatchNorm1d(cfg.REID_HEAD.PERSON_FEATURE.DIM)
        init.constant_(bn_neck.bias, 0.0)
        init.normal_(bn_neck.weight, std=0.01)
        bn_neck.bias.requires_grad_(False)
        res["bn_neck"] = bn_neck
        return res

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            # forward det
            proposals, proposal_losses = self.proposal_generator(
                images, det_bk_features, gt_instances
            )

            # det_pred_instances, det_losses = self.roi_heads.forward_unms(
            #    images, det_bk_features, proposals, gt_instances
            # )
            (
                det_pred_instances,
                det_losses,
                (box_embeds, selected_proposals),
            ) = self.roi_heads(
                images, det_bk_features, proposals, gt_instances, return_embed=True
            )
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    outputs = {"pred_boxes": [], "pred_scores": []}
                    for pi, pred_img in enumerate(det_pred_instances):
                        boxes = pred_img.pred_boxes.tensor
                        scores = pred_img.scores
                        outputs["pred_boxes"].append(boxes)
                        outputs["pred_scores"].append(scores.unsqueeze(1))
                    for gti in gt_instances:
                        gt_id = gti.gt_classes
                        gt_id[gt_id == 0] = -1
                        gt_id[gt_id > 0] -= 2
                    self.visualize_training(
                        (images, gt_instances),
                        det_bk_features[list(det_bk_features.keys())[-1]],
                        outputs,
                    )
            losses = {}
            losses.update(det_losses)
            losses.update(proposal_losses)
            # forward re-id
            # back to original pid
            proposal_ids = torch.cat([p.gt_classes for p in selected_proposals], dim=0)
            proposal_ids[proposal_ids == 0] = -1
            proposal_ids[proposal_ids == 1] = -2
            proposal_ids[proposal_ids > 1] -= 2
            pos_mask = proposal_ids > -2
            pembs = self.bn_neck(box_embeds[pos_mask])
            rl = self.reid_loss(pembs, proposal_ids[pos_mask], None)
            losses.update(rl)
            return losses
        else:
            if self.train_task == "det":
                images, gt_instances = self.preprocess_input(input_list)
                det_bk_features = self.backbone(images.tensor)
                det_pred = self.forward_det(
                    images, det_bk_features, gt_instances
                )  # det eval only
                for pi, bi_boxes in enumerate(det_pred["pred_boxes"]):
                    org_boxes = _resize_boxes(
                        bi_boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                    )
                    det_pred["pred_boxes"][pi] = org_boxes
                return det_pred
            if "query" in input_list[0]:
                return self.inf_query(input_list)
            images, gt_instances = self.preprocess_input(input_list)
            det_bk_features = self.backbone(images.tensor)
            # forward det
            proposals, _ = self.proposal_generator(images, det_bk_features, None)
            det_pred_instances, _, box_embeds = self.roi_heads(
                images, det_bk_features, proposals, None, return_embed=True
            )
            del det_bk_features
            cat_embeds = torch.cat(box_embeds, dim=0)
            reid_feats = self.bn_neck(cat_embeds)
            reid_feats = tF.normalize(reid_feats, dim=-1)
            reid_feats = torch.split(cat_embeds, [boxe.shape[0] for boxe in box_embeds])
            det_pred = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pred_img in enumerate(det_pred_instances):
                boxes = pred_img.pred_boxes.tensor
                scores = pred_img.scores
                det_pred["pred_boxes"].append(boxes)
                det_pred["pred_scores"].append(scores.unsqueeze(1))

            outputs = {"pred_boxes": [], "pred_scores": [], "reid_feats": []}
            for pi, pfeats in enumerate(reid_feats):
                boxes = det_pred["pred_boxes"][pi]
                org_boxes = _resize_boxes(
                    boxes, images.image_sizes[pi], gt_instances[pi]._org_hw
                )
                scores = det_pred["pred_scores"][pi]
                outputs["pred_boxes"].append(org_boxes)
                outputs["pred_scores"].append(scores)
                outputs["reid_feats"].append(pfeats)
            return outputs

    def inf_query(self, input_list):
        images, gt_instances = self.preprocess_input(qd["query"] for qd in input_list)
        det_bk_features = self.backbone(images.tensor)
        q_boxes = [gti.gt_boxes.tensor for gti in gt_instances]
        d2_boxes = [Boxes(pi_boxes) for pi_boxes in q_boxes]
        roi_feats = self.roi_heads._shared_roi_transform(
            [det_bk_features["res4"]], d2_boxes
        )
        del det_bk_features
        q_embs = roi_feats.mean(dim=[2, 3])
        q_embs = self.bn_neck(q_embs)
        q_embs = tF.normalize(q_embs, dim=-1)
        for bi, feat in enumerate(q_embs):
            input_list[bi]["query"]["feat"] = feat
        return input_list


@META_ARCH_REGISTRY.register()
class TiRCNN_C4SideSlr(TiRCNN_C4Side):
    @configurable
    def __init__(self, bn_neck, side_init, **kwargs) -> None:
        super(TiRCNN_C4Side, self).__init__(**kwargs)

        self.side_init = side_init
        self.alpha_res3 = nn.Parameter(torch.tensor(0.0))
        self.alpha_res4 = nn.Parameter(torch.tensor(0.0))
        # side reid network
        self.bn_neck = bn_neck
        self.side_res3 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 4,
                    "stride_per_block": [2, 1, 1, 1],
                    "in_channels": 256,
                    "out_channels": 512,
                    "norm": "BN",
                    "bottleneck_channels": 128,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res4 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 6,
                    "stride_per_block": [2, 1, 1, 1, 1, 1],
                    "in_channels": 512,
                    "out_channels": 1024,
                    "norm": "BN",
                    "bottleneck_channels": 256,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )
        self.side_res5 = nn.Sequential(
            *ResNet.make_stage(
                **{
                    "num_blocks": 3,
                    "stride_per_block": [1, 1, 1],
                    "in_channels": 1024,
                    "out_channels": 2048,
                    "norm": "BN",
                    "bottleneck_channels": 512,
                    "stride_in_1x1": False,
                    "dilation": 1,
                    "num_groups": 1,
                    "block_class": BottleneckBlock,
                }
            )
        )

    def forward(self, input_list):
        if self.training:
            images, gt_instances = self.preprocess_input(input_list)
            if self.train_task == "det":
                det_bk_features = self.backbone(images.tensor)
                return self.forward_det(images, det_bk_features, gt_instances)  # losses
            else:
                det_bk_features = self.backbone(images.tensor)
                det_pred = self.forward_det(images, det_bk_features, gt_instances)
                # back to original pid
                for gti in gt_instances:
                    cur_pids = gti.gt_classes
                    gti.gt_classes[cur_pids == 0] = -1
                    gti.gt_classes[cur_pids > 0] -= 2
                return self.forward_ps(det_pred, det_bk_features, images, gt_instances)
        else:
            return super().forward(input_list)

    def get_reid_backbone_features(self, det_backbone_features, image_list):
        del image_list
        reid_res3_feat = checkpoint.checkpoint(
            self.side_res3, det_backbone_features["res2"]
        )
        alpha = torch.sigmoid(self.alpha_res3)
        reid_res3_feat = (
            alpha * det_backbone_features["res3"] + (1 - alpha) * reid_res3_feat
        )
        reid_res4_feat = checkpoint.checkpoint(self.side_res4, reid_res3_feat)
        del reid_res3_feat
        alpha = torch.sigmoid(self.alpha_res4)
        reid_res4_feat = (
            alpha * det_backbone_features["res4"] + (1 - alpha) * reid_res4_feat
        )
        return reid_res4_feat

    def get_reid_embed(self, p_feat_maps):
        p_feat_maps = checkpoint.checkpoint(
            self.side_res5, p_feat_maps
        )  # n x c x h x w
        pfeat_embs = self.pfeat_pooling(p_feat_maps)
        pfeat_embs = pfeat_embs.view(pfeat_embs.shape[0], pfeat_embs.shape[1])
        pfeat_embs = self.bn_neck(pfeat_embs)
        if self.training:
            return pfeat_embs
        else:
            return tF.normalize(pfeat_embs, dim=-1)


def _load_file(filename):
    from psd2.utils.file_io import PathManager
    import pickle

    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {
                "model": data,
                "__author__": "Caffe2",
                "matching_heuristics": True,
            }
    elif filename.endswith(".pyth"):
        # assume file is from pycls; no one else seems to use the ".pyth" extension
        with PathManager.open(filename, "rb") as f:
            data = torch.load(f)
        assert (
            "model_state" in data
        ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {
            k: v
            for k, v in data["model_state"].items()
            if not k.endswith("num_batches_tracked")
        }
        return {
            "model": model_state,
            "__author__": "pycls",
            "matching_heuristics": True,
        }

    loaded = torch.load(
        filename, map_location=torch.device("cpu")
    )  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded


def _resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
