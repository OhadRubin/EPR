import multiprocessing
import argparse
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
from  qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
import numpy as np
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.qdmr_to_logical_form_tokens import QDMRToQDMRStepTokensConverter

global_state = {}
from qdecomp_with_dependency_graphs.utils.timeout import exit_after
from dataclasses import dataclass


from dataflow.core.lispress import try_round_trip
   
from src.utils import top_utils




def eval_single_mtop(pred: str,gold: str):
    return pred.strip()==gold
    pred_lf = top_utils.deserialize_top(pred)
    gold_lf = top_utils.deserialize_top(gold)
    if pred_lf is None:
        return pred==gold
    else:
        return pred_lf.serialize()==gold_lf.serialize()



def eval_single_smcalflow(pred: str,gold: str):
    return pred.strip()==gold
    pred_lispress = try_round_trip(pred)
    gold_lispress = try_round_trip(gold)
    return pred_lispress==gold_lispress

@dataclass
class GlobalState:
    converter = None
    matcher = None
    scorer = None

    def __post_init__(self):
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()





def eval_single(question, generated,decomposition,index):
    try:
        # print(f"Starting: {index}")
        if "#13" in generated:
            return False
        def try_invoke(func, graph, default=None):
            try:
                return func(graph)
            except Exception as ex:
                return default

        gold = format_qdmr(decomposition)
        pred = format_qdmr(generated.replace("  "," ").lower())
        # pred_graph = pred.to_graph()
        # gold_graph = gold.to_graph()
        # pred_decomp = Decomposition.from_graph(pred)
        # gold_decomp = Decomposition.from_graph(gold)
        # pred_norm_graph = try_invoke(global_state.scorer.normalize_graph, pred_graph, default=pred_graph)
        # gold_norm_graph = try_invoke(global_state.scorer.normalize_graph, gold_graph, default=gold_graph)
        # pred_norm_str = try_invoke(lambda x: Decomposition.from_graph(x).to_string(), pred_norm_graph, default=pred_norm_graph)
        # gold_norm_str = try_invoke(lambda x: Decomposition.from_graph(x).to_string(), gold_norm_graph, default=gold_norm_graph)

        decomp_lf = global_state.converter.convert(question_id=str(index), question_text=question, decomposition=pred.to_break_standard_string())
        gold_lf = global_state.converter.convert(question_id=str(index), question_text=question, decomposition=gold.to_break_standard_string())
        s = global_state.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf, graph2=gold_lf)
        # print(f"{decomposition}\n{generated}\nFinished: {index} | {int(gold_norm_str==pred_norm_str)}")
        # return gold_norm_str==pred_norm_str
        return s
    except Exception as ex:
        # print(f"Failed on: {index} | 0")
        return False


def eval_many(questions, preds,golds,n_proc=None):
    def set_global_object():
        global global_state
        global_state = GlobalState()
        



    pool = multiprocessing.Pool(processes=n_proc,initializer=set_global_object)
    # results = [pool.apply_async(eval_single, args=wd)  for wd in list(zip(preds,golds))]
    mrange = list(range(len(preds)))
    results = pool.starmap_async(eval_single, list(zip(questions, preds,golds,mrange)))
    # results = [p.get(None) for p in results]
    results = results.get(None)

    return results


def eval_many_mtop(preds,golds,n_proc=None):


    pool = multiprocessing.Pool(processes=n_proc)
    # results = [pool.apply_async(eval_single, args=wd)  for wd in list(zip(preds,golds))]
    results = pool.starmap_async(eval_single_mtop, list(zip(preds,golds)))
    # results = [p.get(None) for p in results]
    results = results.get(None)

    return results


def eval_many_smcalflow(preds,golds,n_proc=None):
    pool = multiprocessing.Pool(processes=n_proc)
    # results = [pool.apply_async(eval_single, args=wd)  for wd in list(zip(preds,golds))]
    results = pool.starmap_async(eval_single_smcalflow, list(zip(preds,golds)))
    # results = [p.get(None) for p in results]
    results = results.get(None)

    return results
