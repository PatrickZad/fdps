# Copyright (c) Facebook, Inc. and its affiliates.
from .evaluator import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_context,
    inference_on_dataset,
    inference_vis_on_dataset,
)

from .testing import print_csv_format, verify_results
from .gallery_inf_det_evaluator import InfDetEvaluator
from .query_evaluator import QueryEvaluator

from .prw_query_eval_p import PrwQueryEvaluatorP
from .cuhk_query_eval_p import CuhkQueryEvaluatorP


__all__ = [k for k in globals().keys() if not k.startswith("_")]
