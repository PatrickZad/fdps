import os


import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def tensor_gather(tensor_list):
    """
    Run all_gather on a list of tensors
    (not necessarily same sized in dim0, but should have the same sizes for dim >=1)
    Args:
        tensor_list: List[tensor1, tensor2, ...]
        tensor_list[-1] must be one-dimensional
    Returns:
        list[cat[tensor1s], cat[tensor2s, ...]]: list of tensor gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor_list

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor_list[-1].size()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    lists = [[] for _ in tensor_list]
    for i in range(len(lists)):
        tensor_size = list(tensor_list[i].size())
        tensor_size[0] = max_size
        for _ in size_list:
            lists[i].append(
                torch.empty(tensor_size, dtype=tensor_list[i].dtype, device="cuda")
            )
        if local_size != max_size:
            tensor_size[0] -= local_size
            padding = torch.empty(
                tensor_size, dtype=tensor_list[i].dtype, device="cuda"
            )
            tensor_list[i] = torch.cat((tensor_list[i], padding), dim=0)

        dist.all_gather(lists[i], tensor_list[i])

        for j, real_size in enumerate(size_list):
            lists[i][j] = lists[i][j][:real_size, ...]
        lists[i] = torch.cat(lists[i], dim=0)

    return lists


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
