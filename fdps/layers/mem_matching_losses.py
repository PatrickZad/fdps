from turtle import forward
import torch
import torch.nn as nn
from torch.autograd import Function
from fdps.utils import comm
import functools
import torch.nn.functional as F


def _lb_feat_update(lookup_table, labeled_feats, labeled_ids, labeled_mms, do_norm):
    for indx, label in enumerate(labeled_ids):
        if labeled_mms[indx] < 1:
            lookup_table[label] = (
                labeled_mms[indx] * lookup_table[label]
                + (1 - labeled_mms[indx]) * labeled_feats[indx]
            )
            if do_norm:
                lookup_table[label] /= lookup_table[label].norm()


def _lb_feat_update_wavg(
    lookup_table, labeled_feats, labeled_ids, labeled_mms, do_norm
):
    id_set = list(set(labeled_ids.tolist()))
    for pid in id_set:
        id_mask = labeled_ids == pid
        p_feats = labeled_feats[id_mask]
        p_mms = labeled_mms[id_mask].unsqueeze(1)
        p_ws = 1 - p_mms
        w_feat = (p_ws * p_feats).sum(dim=0) / p_ws.sum()
        mm = 1 - p_ws.mean()
        lookup_table[pid] = mm * lookup_table[pid] + (1 - mm) * w_feat
        if do_norm:
            lookup_table[pid] /= lookup_table[pid].norm()


def _ulb_feat_update(queue, tail, unlabeled_feats, length):
    if unlabeled_feats.shape[0] > queue.shape[0]:
        unlabeled_feats = unlabeled_feats[-queue.shape[0] :]
    num_feats = unlabeled_feats.shape[0]
    tail_v = tail[0]
    if num_feats <= queue.shape[0] - tail_v:
        queue[tail_v : tail_v + num_feats, :length] = unlabeled_feats[:, :length]
    else:
        num_left = num_feats - (queue.shape[0] - tail_v)
        queue[tail_v:, :length] = unlabeled_feats[:-num_left, :length]
        queue[:num_left, :length] = unlabeled_feats[-num_left:, :length]


def _update_table_sync(lookup_table, pid_labels, features, momentums, feat_update_func):
    # Update lookup table, but not by standard backpropagation with gradients
    labeled_mask = pid_labels > -1
    labeled_feats = features[labeled_mask]
    labeled_ids = pid_labels[labeled_mask]
    labeled_mms = momentums[labeled_mask]
    labeled_tuple = (labeled_feats, labeled_ids, labeled_mms)
    comm.synchronize()
    all_labeled_tuples = comm.all_gather(labeled_tuple)
    all_labeled_feats = []
    all_labeled_ids = []
    all_labeled_mms = []
    for feats, ids, mms in all_labeled_tuples:
        if feats.shape[0] == 0:
            continue
        all_labeled_feats.append(feats.to(labeled_feats.device))
        all_labeled_ids.append(ids.to(labeled_ids.device))
        all_labeled_mms.append(mms.to(labeled_mms.device))
    if len(all_labeled_feats) > 0:
        cat_labeled_feats = torch.cat(all_labeled_feats, dim=0)
        cat_labeled_ids = torch.cat(all_labeled_ids, dim=0)
        cat_labeled_mms = torch.cat(all_labeled_mms, dim=0)
        feat_update_func(
            lookup_table, cat_labeled_feats, cat_labeled_ids, cat_labeled_mms
        )


def _update_table_usync(
    lookup_table, pid_labels, features, momentums, feat_update_func
):
    # Update lookup table, but not by standard backpropagation with gradients
    labeled_mask = pid_labels > -1
    labeled_feats = features[labeled_mask]
    labeled_ids = pid_labels[labeled_mask]
    labeled_mms = momentums[labeled_mask]
    feat_update_func(lookup_table, labeled_feats, labeled_ids, labeled_mms)


def _update_queue_sync(queue, tail, pid_labels, features, uplen):
    # Update circular queue, but not by standard backpropagation with gradients
    unlabeled_mask = pid_labels == -1
    unlabeled_feats = features[unlabeled_mask]
    comm.synchronize()
    all_unlabeled_feats = comm.all_gather(unlabeled_feats)
    all_unlabeled_feats = [
        feats.to(unlabeled_feats.device)
        for feats in all_unlabeled_feats
        if feats.shape[0] > 0
    ]
    if len(all_unlabeled_feats) > 0:
        all_unlabeled_feats = torch.cat(all_unlabeled_feats, dim=0)
        _ulb_feat_update(queue, tail, all_unlabeled_feats, uplen)
        tail[0] = (tail[0] + all_unlabeled_feats.shape[0]) % queue.shape[0]


def _update_queue_usync(queue, tail, pid_labels, features, uplen):
    # Update circular queue, but not by standard backpropagation with gradients
    unlabeled_mask = pid_labels == -1
    unlabeled_feats = features[unlabeled_mask]
    if unlabeled_feats.shape[0] > 0:
        _ulb_feat_update(queue, tail, unlabeled_feats, uplen)
        tail = (tail + unlabeled_feats.shape[0]) % queue.shape[0]


class LabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pid_labels, lookup_table, momentum):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        scores = features.mm(lookup_table.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingNorm(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingNormUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayer(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_persons=5532, feat_len=256, sync=True):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(LabeledMatchingLayer, self).__init__()
        self.register_buffer("lookup_table", torch.zeros(num_persons, feat_len))
        self.sync = sync

    def _scores_sync(self, features, pids, mms):
        return LabeledMatching.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingUsync.apply(features, pids, self.lookup_table, mms)

    def forward(self, features, pid_labels, momentums=None):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        if momentums is None:
            n_feats = features.shape[0]
            momentums = features.new_zeros(n_feats) + 0.5
        if self.sync:
            scores = self._scores_sync(features, pid_labels, momentums)
        else:
            scores = self._scores_usync(features, pid_labels, momentums)

        return scores


class LabeledMatchingLayerNorm(LabeledMatchingLayer):
    """
    Labeled matching of OIM loss function.
    """

    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingNorm.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingNormUsync.apply(features, pids, self.lookup_table, mms)


class LabeledMatchingLayerNml(LabeledMatchingLayer):
    def _scores_sync(self, features, pids, mms):
        features = F.normalize(features, dim=1)
        return LabeledMatchingNorm.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        features = F.normalize(features, dim=1)
        return LabeledMatchingNormUsync.apply(features, pids, self.lookup_table, mms)


class LabeledMatchingWavg(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_wavg, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingWavgUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_wavg, do_norm=False),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayerWavg(LabeledMatchingLayer):
    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingWavg.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingWavgUsync.apply(features, pids, self.lookup_table, mms)


class LabeledMatchingNormWavg(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_sync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_wavg, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingNormWavgUsync(LabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)
        _update_table_usync(
            lookup_table,
            pid_labels,
            features,
            momentum,
            feat_update_func=functools.partial(_lb_feat_update_wavg, do_norm=True),
        )
        return grad_feats, None, None, None


class LabeledMatchingLayerNormWavg(LabeledMatchingLayerWavg):
    def _scores_sync(self, features, pids, mms):
        return LabeledMatchingNormWavg.apply(features, pids, self.lookup_table, mms)

    def _scores_usync(self, features, pids, mms):
        return LabeledMatchingNormWavgUsync.apply(
            features, pids, self.lookup_table, mms
        )


class UnlabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pid_labels, queue, tail):
        # The queue/tail can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pid_labels)
        ctx.queue = queue
        ctx.tail = tail

        scores = features.mm(queue.t())
        return scores

    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)
        # print(tail)
        _update_queue_sync(queue, tail, pid_labels, features, 64)
        return grad_feats, None, None, None


class UnlabeledMatchingUsync(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_usync(queue, tail, pid_labels, features, 64)
        return grad_feats, None, None, None


class UnlabeledMatchingLayer(nn.Module):
    """
    Unlabeled matching of OIM loss function.
    """

    def __init__(self, queue_size=5000, feat_len=256, sync=True):
        """
        Args:
            queue_size (int): Size of the queue saving the features of unlabeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(UnlabeledMatchingLayer, self).__init__()
        self.register_buffer("queue", torch.zeros(queue_size, feat_len))
        self.register_buffer("tail", torch.tensor([0]))
        # self.tail = 0
        self.sync = sync

    def _scores_sync(self, features, pids):
        return UnlabeledMatching.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingUsync.apply(features, pids, self.queue, self.tail)

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, queue_size]): Unlabeled matching scores, namely the similarities
                                            between proposals and unlabeled persons.
        """
        if self.sync:
            scores = self._scores_sync(features, pid_labels)
        else:
            scores = self._scores_usync(features, pid_labels)
        return scores


class UnlabeledMatchingLayerNml(UnlabeledMatchingLayer):
    def _scores_sync(self, features, pids):
        features = F.normalize(features, dim=1)
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        features = F.normalize(features, dim=1)
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


class UnlabeledMatchingFull(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_sync(queue, tail, pid_labels, features, features.shape[1])
        return grad_feats, None, None, None


class UnlabeledMatchingFullUsync(UnlabeledMatching):
    @staticmethod
    def backward(ctx, grad_output):
        features, pid_labels = ctx.saved_tensors
        queue = ctx.queue
        tail = ctx.tail

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(queue.data)

        _update_queue_usync(queue, tail, pid_labels, features, features.shape[1])
        return grad_feats, None, None, None

    def _scores_sync(self, features, pids):
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


class UnlabeledMatchingFullLayer(UnlabeledMatchingLayer):
    def _scores_sync(self, features, pids):
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


class UnlabeledMatchingFullLayerNml(UnlabeledMatchingLayer):
    def _scores_sync(self, features, pids):
        features = F.normalize(features, dim=1)
        return UnlabeledMatchingFull.apply(features, pids, self.queue, self.tail)

    def _scores_usync(self, features, pids):
        features = F.normalize(features, dim=1)
        return UnlabeledMatchingFullUsync.apply(features, pids, self.queue, self.tail)


class LabeledMatchingEmaLayer(nn.Module):
    def __init__(self, ql, feat_dim=256, sync=True):
        super().__init__()
        self.register_buffer("feat_queue", torch.zeros(ql, feat_dim))
        self.register_buffer("id_queue", torch.zeros(ql, dtype=torch.int32) - 2)
        self.sync = sync
        self.next_idx = 0

    def forward(self, pred_feats, ema_feats, ema_ids):
        if self.sync:
            cur_emas = (ema_feats, ema_ids)
            comm.synchronize()
            all_emas = comm.all_gather(cur_emas)
            cur_ema_feats = []
            cur_ema_ids = []
            for emas in all_emas:
                cur_ema_feats.append(emas[0].to(self.feat_queue.device))
                cur_ema_ids.append(emas[1].to(self.feat_queue.device))
            ema_feats = torch.cat(cur_ema_feats, dim=0)
            ema_ids = torch.cat(cur_ema_ids, dim=0)
        self.enqueue(ema_feats, ema_ids)
        return torch.mm(pred_feats, self.feat_queue.t())

    def get_buffer(self, pred_feats, ema_feats, ema_ids):
        if self.sync:
            cur_emas = (ema_feats, ema_ids)
            comm.synchronize()
            all_emas = comm.all_gather(cur_emas)
            cur_ema_feats = []
            cur_ema_ids = []
            for emas in all_emas:
                cur_ema_feats.append(emas[0].to(self.feat_queue.device))
                cur_ema_ids.append(emas[1].to(self.feat_queue.device))
            ema_feats = torch.cat(cur_ema_feats, dim=0)
            ema_ids = torch.cat(cur_ema_ids, dim=0)
        self.enqueue(ema_feats, ema_ids)
        mask = self.id_queue > -2
        return self.feat_queue[mask], self.id_queue[mask]

    def enqueue(self, feats, ids):
        num_feats = feats.shape[0]
        if num_feats <= self.feat_queue.shape[0] - self.next_idx:
            self.feat_queue[self.next_idx : self.next_idx + num_feats] = feats
            self.id_queue[self.next_idx : self.next_idx + num_feats] = ids
        else:
            num_left = num_feats - (self.feat_queue.shape[0] - self.next_idx)
            self.feat_queue[self.next_idx :] = feats[:-num_left]
            self.id_queue[self.next_idx :] = ids[:-num_left]
            self.feat_queue[:num_left] = feats[-num_left:]
            self.id_queue[:num_left] = ids[-num_left:]
        self.next_idx = (self.next_idx + num_feats) % self.feat_queue.shape[0]


class UnlabeledMatchingEmaLayer(LabeledMatchingEmaLayer):
    pass


"""
For weakly supervised setting
Default update operation is the same as labled in OIM
"""


class MemoryBasedMatchingLayer(LabeledMatchingLayerNorm):
    def update_mem(self, new_mem):
        self.lookup_table = new_mem


layers_map = {
    "lb": LabeledMatchingLayer,
    "lb_norm": LabeledMatchingLayerNorm,
    "lb_nml": LabeledMatchingLayerNml,
    "lb_wavg": LabeledMatchingLayerNormWavg,
    "ulb": UnlabeledMatchingLayer,
    "ulb_nml": UnlabeledMatchingLayerNml,
    "ulb_full": UnlabeledMatchingFullLayer,
    "ulb_full_nml": UnlabeledMatchingFullLayerNml,
    "lb_ema": LabeledMatchingEmaLayer,
    "ulb_ema": UnlabeledMatchingEmaLayer,
    "mem_matching": MemoryBasedMatchingLayer,
}



class OIMLoss(nn.Module):
    def __init__(
        self,
        lb_layer,
        ulb_layer,
        lb_factor,
        ulb_factor,
        num_lb,
        num_ulb,
        feat_len,
        loss_weights,
        normalize=True,
        sync=True,
        use_focal=True,
        focal_alpha=1,
        focal_gamma=2,
    ):
        super().__init__()
        self.lb_layer = layers_map[lb_layer](num_lb, feat_len, sync)
        if num_ulb == 0:
            self.ulb_layer = None
        else:
            self.ulb_layer = layers_map[ulb_layer](num_ulb, feat_len, sync)
        self.lb_factor = lb_factor
        self.ulb_factor = ulb_factor
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_weights = loss_weights
        self.normalize = normalize

    def forward(self, pfeats, pids, lb_mms):
        if self.normalize:
            pfeats = F.normalize(pfeats, dim=-1)
        lb_matching_scores = self.lb_layer(pfeats, pids, lb_mms) * self.lb_factor
        if self.ulb_layer:
            ulb_matching_scores = self.ulb_layer(pfeats, pids) * self.ulb_factor
            matching_scores = torch.cat(
                (lb_matching_scores, ulb_matching_scores), dim=1
            )
        else:
            matching_scores = lb_matching_scores
        pid_labels = pids.clone()
        pid_labels[pid_labels == -2] = -1
        n_lb_feats = (pid_labels > -1).sum()
        if lb_matching_scores.shape[0] == 0:
            loss_oim = lb_matching_scores.new_tensor([0])
        else:
            if self.use_focal:
                p_i = F.softmax(matching_scores, dim=1)
                focal_p_i = self.focal_alpha * (1 - p_i) ** self.focal_gamma * p_i.log()
                loss_oim = F.nll_loss(
                    focal_p_i, pid_labels, reduction="none", ignore_index=-1
                )
            else:
                loss_oim = F.cross_entropy(
                    matching_scores, pid_labels, reduction="none", ignore_index=-1
                )
        comm.synchronize()
        all_num_lb = comm.all_gather(n_lb_feats)
        num_lb = sum([num.to("cpu") for num in all_num_lb])
        num_lb = torch.clamp(num_lb / comm.get_world_size(), min=1).item()
        loss_val = loss_oim.sum() / num_lb
        return {"loss_oim": loss_val * self.loss_weights["loss_oim"]}


class OIMTridLoss(OIMLoss):
    def __init__(
        self,
        lb_layer,
        ulb_layer,
        lb_factor,
        ulb_factor,
        bin_factor,
        num_lb,
        num_ulb,
        feat_len,
        loss_weights,
        normalize=True,
        sync=True,
        use_focal=False,
        focal_alpha=1,
        focal_gamma=2,
    ):
        super().__init__(
            lb_layer,
            ulb_layer,
            lb_factor,
            ulb_factor,
            num_lb,
            num_ulb,
            feat_len,
            loss_weights,
            normalize=normalize,
            sync=sync,
            use_focal=use_focal,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        self.bin_factor = bin_factor

    def forward(self, pfeats, pids, lb_mms):
        if self.normalize:
            pfeats = F.normalize(pfeats, dim=-1)
        lb_matching_scores = self.lb_layer(pfeats, pids, lb_mms)
        ulb_matching_scores = self.ulb_layer(pfeats, pids)
        lb_mask = pids > -1
        ulb_mask = pids == -1
        n_labeled = lb_mask.sum()
        n_unlabeled = ulb_mask.sum()
        b_ids = lb_matching_scores.new_tensor(list(range(n_labeled)), dtype=torch.long)
        lb_gt_matching_scores = lb_matching_scores[lb_mask][(b_ids, pids[lb_mask])]
        sync_data = (n_labeled, n_unlabeled, lb_gt_matching_scores.detach())
        comm.synchronize()
        all_num_lbulb_lbgt = comm.all_gather(sync_data)
        sum_lb, sum_ulb = 0, 0
        lb_gt_mt_scores = []
        for nlb, nulb, mt_scores in all_num_lbulb_lbgt:
            sum_lb += nlb.cpu()
            sum_ulb += nulb.cpu()
            lb_gt_mt_scores.append(mt_scores.to(lb_gt_matching_scores.device))
        num_lb = torch.clamp(sum_lb / comm.get_world_size(), min=1).item()
        num_ulb = torch.clamp(sum_ulb / comm.get_world_size(), min=1).item()
        # all_lb_gt_mt_scores = torch.cat(lb_gt_mt_scores, dim=0)
        # loss bin
        if lb_gt_matching_scores.shape[0] == 0:
            loss_bin = lb_gt_matching_scores.new_tensor([0])
        else:
            loss_bin = (
                1 + ((1 -  lb_gt_matching_scores) * self.bin_factor).exp()
            ).log()  # (num_lb, )
        # loss unlabeled
        ulb_lb_matching_scores = lb_matching_scores[ulb_mask]
        if ulb_lb_matching_scores.shape[0] == 0 or lb_gt_matching_scores.shape[0] == 0:
            loss_un = ulb_lb_matching_scores.new_tensor([0])
        else:
            if comm.get_world_size()==1:
                all_mean_lb_gt_mt_scores = lb_gt_mt_scores[0].sum()/ lb_gt_mt_scores[0].shape[0]
            else:
                this_rank = comm.get_rank()
                other_lb_gt_mt_scores = torch.cat(
                    lb_gt_mt_scores[:this_rank] + lb_gt_mt_scores[this_rank + 1 :], dim=0
                )
                all_mean_lb_gt_mt_scores = (
                    lb_gt_mt_scores.sum() + other_lb_gt_mt_scores.sum()
                ) / (lb_gt_mt_scores.shape[0] + other_lb_gt_mt_scores.shape[0])
            loss_un = (
                1
                + (
                    (ulb_lb_matching_scores - all_mean_lb_gt_mt_scores)
                    * self.ulb_factor
                )
                .exp()
                .sum(-1)
            ).log()  # (num_ulb, )
        # loss id
        id_lb_matching_scores = lb_matching_scores * self.lb_factor
        id_ulb_matching_scores = ulb_matching_scores * self.ulb_factor
        id_matching_scores = torch.cat(
            (id_lb_matching_scores, id_ulb_matching_scores), dim=1
        )  # + MINI
        pid_labels = pids.clone()
        pid_labels[pid_labels == -2] = -1
        if lb_matching_scores.shape[0] == 0:
            loss_oim = lb_matching_scores.new_tensor([0])
        else:
            if self.use_focal:
                p_i = F.softmax(id_matching_scores, dim=1)
                focal_p_i = self.focal_alpha*(1 - p_i) ** self.focal_gamma * p_i.log()
                loss_oim = F.nll_loss(
                    focal_p_i, pid_labels, reduction="none", ignore_index=-1
                )
            else:
                loss_oim = F.cross_entropy(
                    id_matching_scores, reduction="none", ignore_index=-1
                )
        return {
            "loss_oim": loss_oim.sum() / num_lb * self.loss_weights["loss_oim"],
            "loss_bin": loss_bin.sum() / num_lb * self.loss_weights["loss_bin"],
            "loss_un": loss_un.sum() / num_ulb * self.loss_weights["loss_un"],
        }


def build_loss_layer(loss_cfg, feat_dim):
    if loss_cfg.NAME == "oim":
        oim_cfg = loss_cfg.OIM
        loss_weights = {"loss_oim": loss_cfg.LOSS_WEIGHTS.OIM}
        return OIMLoss(
            lb_layer=oim_cfg.LB_LAYER,
            ulb_layer=oim_cfg.ULB_LAYER,
            lb_factor=oim_cfg.LB_FACTOR,
            ulb_factor=oim_cfg.ULB_FACTOR,
            num_lb=oim_cfg.LUT_LEN,
            num_ulb=oim_cfg.QUEUE_LEN,
            feat_len=feat_dim,
            loss_weights=loss_weights,
            normalize=oim_cfg.NORMALIZE,
            sync=oim_cfg.SYNC,
            use_focal=oim_cfg.FOCAL.USE_FOCAL,
            focal_alpha=oim_cfg.FOCAL.FOCAL_ALPHA,
            focal_gamma=oim_cfg.FOCAL.FOCAL_GAMMA,
        )
    elif loss_cfg.NAME == "oim_trid":
        oim_cfg = loss_cfg.OIM
        loss_weights = {
            "loss_oim": loss_cfg.LOSS_WEIGHTS.OIM,
            "loss_bin": loss_cfg.LOSS_WEIGHTS.BIN,
            "loss_un": loss_cfg.LOSS_WEIGHTS.ULB,
        }
        return OIMTridLoss(
            lb_layer=oim_cfg.LB_LAYER,
            ulb_layer=oim_cfg.ULB_LAYER,
            lb_factor=oim_cfg.LB_FACTOR,
            ulb_factor=oim_cfg.ULB_FACTOR,
            bin_factor=oim_cfg.BIN_FACTOR,
            num_lb=oim_cfg.LUT_LEN,
            num_ulb=oim_cfg.QUEUE_LEN,
            feat_len=feat_dim,
            loss_weights=loss_weights,
            normalize=oim_cfg.NORMALIZE,
            sync=oim_cfg.SYNC,
            use_focal=oim_cfg.FOCAL.USE_FOCAL,
            focal_alpha=oim_cfg.FOCAL.FOCAL_ALPHA,
            focal_gamma=oim_cfg.FOCAL.FOCAL_GAMMA,
        )
