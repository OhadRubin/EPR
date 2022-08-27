"""
Evaluate QDMR predictions file
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numbers
from itertools import zip_longest
from enum import Enum

import argparse
import os
import random
import re
import json
import numpy as np
import pandas as pd

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.qdmr_to_logical_form_tokens import QDMRToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.evaluation.graph_matcher import GraphMatchScorer, get_ged_plus_scores
from qdecomp_with_dependency_graphs.evaluation.sari_hook import get_sari
from qdecomp_with_dependency_graphs.evaluation.sequence_matcher import SequenceMatchScorer
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
import qdecomp_with_dependency_graphs.evaluation.normal_form.normalization_rules as norm_rules


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


#############################################
#              Evaluate                     #
#############################################

class EvalMatrics(str, Enum):
    EXACT_MATCH, MATCH, STRUCTURAL_MATCH, SARI, \
    GED, STRUCTURAL_GED, GED_PLUS,\
    NORMALIZED_EXACT_MATCH, NORMALIZED_MATCH, NORMALIZED_STRUCTURAL_MATCH, NORMALIZED_SARI, \
    LOGICAL_FORM_EXACT_MATCH \
    =\
    'exact_match', 'match', 'structural_match', 'sari', \
    'ged', 'structural_ged', 'ged_plus', \
    'normalized_exact_match', 'normalized_match', 'normalized_structural_match', 'normalized_sari', \
    'logical_form_em'




def evaluate(question_ids, questions, decompositions: List[Decomposition], golds: List[Decomposition], metadata,
             output_path_base, num_processes,
             metrics: List[EvalMatrics] = [
                 EvalMatrics.EXACT_MATCH, EvalMatrics.MATCH, EvalMatrics.STRUCTURAL_MATCH, EvalMatrics.SARI,
                 EvalMatrics.NORMALIZED_EXACT_MATCH, EvalMatrics.NORMALIZED_MATCH,
                 EvalMatrics.NORMALIZED_STRUCTURAL_MATCH, EvalMatrics.NORMALIZED_SARI,
                 EvalMatrics.LOGICAL_FORM_EXACT_MATCH,
             ]
             ):
    decompositions_str = [d.to_string() for d in decompositions]
    golds_str = [g.to_string() for g in golds]

    evaluation_dict = {
        "question_id": question_ids,
        "question": questions,
        "gold": golds_str,
        "prediction": decompositions_str,
    }

    # calculating exact match scores
    if (metrics is None) or EvalMatrics.EXACT_MATCH in metrics:
        evaluation_dict[EvalMatrics.EXACT_MATCH.value] = get_exact_match(decompositions_str, golds_str)

    # evaluate using SARI
    if (metrics is None) or EvalMatrics.SARI in metrics:
        evaluation_dict[EvalMatrics.SARI.value] = get_sari_score(decompositions_str, golds_str, questions)

    # evaluate using sequence matcher
    if (metrics is None) or EvalMatrics.MATCH in metrics:
        evaluation_dict[EvalMatrics.MATCH.value] = get_match_ratio(decompositions_str, golds_str)
    if (metrics is None) or EvalMatrics.STRUCTURAL_MATCH in metrics:
        evaluation_dict[EvalMatrics.STRUCTURAL_MATCH.value] = get_structural_match_ratio(decompositions_str, golds_str)

    # evaluate using graph distances
    graph_scorer = GraphMatchScorer()
    decomposition_graphs = [d.to_graph() for d in decompositions]
    gold_graphs = [g.to_graph() for g in golds]

    if (metrics is None) or EvalMatrics.GED in metrics:
        evaluation_dict[EvalMatrics.GED.value] = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)
    if (metrics is None) or EvalMatrics.STRUCTURAL_GED in metrics:
        evaluation_dict[EvalMatrics.STRUCTURAL_GED.value] = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs, structure_only=True)
    if (metrics is None) or EvalMatrics.GED_PLUS in metrics:
        evaluation_dict[EvalMatrics.GED_PLUS.value] = get_ged_plus_scores(decomposition_graphs, gold_graphs,
                                          exclude_thr=5, num_processes=num_processes)

    # calculate normalized match scores
    if any([(x in metrics) for x in [
        EvalMatrics.NORMALIZED_EXACT_MATCH, EvalMatrics.NORMALIZED_MATCH,
        EvalMatrics.NORMALIZED_STRUCTURAL_MATCH, EvalMatrics.NORMALIZED_SARI]]):

        normalize_scorer = NormalizedGraphMatchScorer()

        def try_invoke(func, graph, default=None):
            try:
                return func(graph)
            except Exception as ex:
                return default
        decomposition_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in
                                     decomposition_graphs]
        decomposition_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in
                                  decomposition_norm_graphs]
        gold_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in gold_graphs]
        gold_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in gold_norm_graphs]

        if (metrics is None) or EvalMatrics.NORMALIZED_EXACT_MATCH in metrics:
            evaluation_dict[EvalMatrics.NORMALIZED_EXACT_MATCH.value] = skip_none(get_exact_match, decomposition_norm_str, gold_norm_str)
        if (metrics is None) or EvalMatrics.NORMALIZED_SARI in metrics:
            evaluation_dict[EvalMatrics.NORMALIZED_SARI.value] = skip_none(get_sari_score, decomposition_norm_str, gold_norm_str, questions)
        if (metrics is None) or EvalMatrics.NORMALIZED_MATCH in metrics:
            evaluation_dict[EvalMatrics.NORMALIZED_MATCH.value] = skip_none(get_match_ratio, decomposition_norm_str, gold_norm_str)
        if (metrics is None) or EvalMatrics.NORMALIZED_STRUCTURAL_MATCH in metrics:
            evaluation_dict[EvalMatrics.NORMALIZED_STRUCTURAL_MATCH.value] = skip_none(get_structural_match_ratio, decomposition_norm_str, gold_norm_str)

    # logical form
    if (metrics is None) or EvalMatrics.LOGICAL_FORM_EXACT_MATCH in metrics:
        evaluation_dict[EvalMatrics.LOGICAL_FORM_EXACT_MATCH.value] = get_logical_form_em(question_ids, questions, decompositions, golds)

    num_examples = len(questions)
    print_first_example_scores(evaluation_dict, min(5, num_examples))
    mean_scores = print_score_stats(evaluation_dict)

    if output_path_base:
        write_evaluation_output(output_path_base, mean_scores, **evaluation_dict)

    if metadata is not None:
        #metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
        metadata = metadata[metadata['question_id'].isin(evaluation_dict['question_id'])].copy()
        metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
        if "dataset" not in metadata:
            metadata['dataset'] = metadata['question_id'].apply(lambda x: x.split('_')[0])

        score_keys = [key for key in evaluation_dict if key not in ["question_id", "question", "gold", "prediction"]]
        for key in score_keys:
            metadata[key] = evaluation_dict[key]
            metadata[key] = metadata[key].astype(float)  # deal with boolean missing values

        for agg_field in ["dataset", "num_steps"]:
            df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
            print(df.round(decimals=3))

    return mean_scores


def skip_none(func, *args, **kwargs):
    zipped = list(zip_longest(*args))
    none_ids = [i for i, x in enumerate(zipped) if None in x]
    args_ = tuple([x for i,x in enumerate(a) if i not in none_ids] for a in args)
    res = func(*args_, **kwargs)

    combined = []
    none_i = 0
    res_i = 0
    for i in range(len(zipped)):
        if none_i < len(none_ids) and (i == none_ids[none_i]):
            combined.append(None)
            none_i += 1
        else:
            combined.append(res[res_i])
            res_i += 1
    return combined


#############################################
#              Calculate Scores             #
#############################################

def get_exact_match(decompositions_str:[str], golds_str:[str]):
    return [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]


def get_sari_score(decompositions_str: [str], golds_str: [str], questions: [str]):
    sources = [q.split(" ") for q in questions]
    predictions = [d.split(" ") for d in decompositions_str]
    targets = [[g.split(" ")] for g in golds_str]
    sari, keep, add, deletion = get_sari(sources, predictions, targets)
    return sari


def get_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="base")


def get_structural_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="structural")


def get_logical_form_em(question_ids: [str], question_texts:[ str], decompositions:List[Decomposition], golds:List[Decomposition]):
    converter = QDMRToQDMRStepTokensConverter()
    matcher = LogicalFromStructuralMatcher()
    scores = []
    for qid, q, decomp, gold in zip(question_ids, question_texts, decompositions, golds):
        s = False
        try:
            decomp_lf = converter.convert(question_id=qid, question_text=q, decomposition=decomp.to_break_standard_string())
            gold_lf = converter.convert(question_id=qid, question_text=q, decomposition=gold.to_break_standard_string())
            s = matcher.is_match(question_id=qid, question_text=q, graph1=decomp_lf, graph2=gold_lf)
        except Exception as ex:
            pass
        scores.append(s)
    return scores

#############################################
#              Print & Format               #
#############################################

def print_first_example_scores(evaluation_dict, num_examples):
    for i in range(num_examples):
        print("evaluating example #{}".format(i))
        for k,v in evaluation_dict.items():
            if isinstance(v[i], numbers.Number):
                print("\t{}: {}".format(k, round(v[i], 3)))
            else:
                print("\t{}: {}".format(k, v[i]))


def print_score_stats(evaluation_dict):
    skiped_samples = {}
    mean_scores = {}

    print("\noverall scores:")
    for key in evaluation_dict:
        # ignore keys that do not store scores
        if key in ["question_id", "question", "gold", "prediction"]:
            continue
        score_name, scores = key, evaluation_dict[key]

        # ignore examples without a score
        if None in scores:
            scores_ = [score for score in scores if score is not None]
            skiped_samples[key] = len(scores)-len(scores_)
        else:
            scores_ = scores

        mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
        print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
            score_name, mean_score, max_score, min_score))
        mean_scores[score_name] = mean_score

    for score, skiped in skiped_samples.items():
        print(f"skipped {skiped} examples when computing {score}.")

    return mean_scores


def write_evaluation_output(output_path_base, mean_scores, **kwargs):
    # write evaluation summary
    with open(output_path_base + '_summary.json', 'w') as fd:
        json.dump(mean_scores, fd, indent=2, sort_keys=True)

    # write evaluation scores per example
    df = pd.DataFrame.from_dict(kwargs, orient="columns")
    df.to_csv(output_path_base + '_full.csv', index=False)


def format_qdmr(input:str) -> Decomposition:
    # replace multiple whitespaces with a single whitespace.
    input = ' '.join(input.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = input.split(';')
    parts = [re.sub(r'return', '', part.strip().strip('\r')) for part in parts]

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    parts = [re.sub(r'#(\d+)', '@@\g<1>@@', part) for part in parts]

    return Decomposition(parts)


def get_predictions_from_allennlp_preds_file(predictions_file: str, format: bool = True):
    with open(predictions_file, "r") as fd:
        preds_rows = [json.loads(line) for line in fd.readlines()]

    def get_prediction_tokens(dict):
        if dict['predicted_tokens'] and isinstance(dict['predicted_tokens'][0], list):
            return dict['predicted_tokens'][0]
        return dict['predicted_tokens']

    def format_prediction(text: str):
        if not format:
            return text
        return re.sub(r'@@(\d+)@@', '#\g<1>', re.sub('@@(SEP|sep)@@', ';', text))

    preds = [format_prediction(' '.join(get_prediction_tokens(p))) for p in preds_rows]
    ids = [x['metadata']['question_id'] for x in preds_rows if ('metadata' in x and 'question_id' in x['metadata'])]
    return preds, ids if len(ids)==len(preds) else None


def allennlp_to_official_csv(predictions_file: str =None, format: bool = True, root_dir: str = None):
    assert predictions_file or root_dir
    if root_dir:
        pathlist = [str(x) for x in Path(root_dir).rglob("*test_seq2seq*_preds.json")]
    else:
        pathlist = [predictions_file]
    for path in pathlist:
        try:
            print(f"process {path}")
            preds, ids = get_predictions_from_allennlp_preds_file(predictions_file=predictions_file, format=format)
            df = pd.DataFrame.from_dict({'question_id':ids, 'decomposition': preds})
            df.to_csv(os.path.splitext(predictions_file)[0]+'__official.csv', index=False)
        except Exception as ex:
            print("Error")


def evaluate_predictions(dataset_file:str, preds_file:str, output_file_base:str,
         allennlp:bool = False, random_n:int = None, num_processes:int=5, use_cache:bool=True,
         metrics:List[EvalMatrics] = [
                 EvalMatrics.EXACT_MATCH, EvalMatrics.MATCH, EvalMatrics.STRUCTURAL_MATCH, EvalMatrics.SARI,
                 EvalMatrics.NORMALIZED_EXACT_MATCH, EvalMatrics.NORMALIZED_MATCH,
                 EvalMatrics.NORMALIZED_STRUCTURAL_MATCH, EvalMatrics.NORMALIZED_SARI,
                 EvalMatrics.LOGICAL_FORM_EXACT_MATCH,
             ]):
    # load data
    try:
        metadata = pd.read_csv(dataset_file)
        ids = metadata["question_id"].to_list()
        questions = metadata["question_text"].to_list()
        golds = [format_qdmr(decomp) for decomp in metadata["decomposition"].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load dataset file {dataset_file}", ex)

    # load predictions
    try:
        if allennlp:
            predictions, question_ids = get_predictions_from_allennlp_preds_file(preds_file)
            if len(predictions) != len(ids):
                if question_ids:
                    ids_to_index = {x:i for i,x in enumerate(ids)}
                    indices = [ids_to_index[x] for x in question_ids]
                    ids = [ids[i] for i in indices]
                    questions = [questions[i] for i in indices]
                    golds = [golds[i] for i in indices]
                    predictions = [predictions[i] for i in indices]
                else:
                    raise ValueError(f"mismatch number of predictions ({len(predictions)}) and dataset ({len(ids)}). "
                                     f"no question_id field was found in predictions metadata")
        else:
            preds_df = pd.read_csv(preds_file)
            predictions = preds_df['decomposition'].where(pd.notnull(preds_df['decomposition']), '').to_list()
        predictions = [format_qdmr(pred) for pred in predictions]
    except Exception as ex:
        raise ValueError(f"Could not load predictions file {preds_file}", ex)

    assert len(golds) == len(predictions), "mismatch number of gold questions and predictions"

    if random_n and len(golds) > random_n:
        indices = random.sample(range(len(ids)), random_n)
        ids = [ids[i] for i in indices]
        questions = [questions[i] for i in indices]
        golds = [golds[i] for i in indices]
        predictions = [predictions[i] for i in indices]

    if use_cache:
        norm_rules.load_cache(os.path.splitext(dataset_file)[0]+"__cache")

    res = evaluate(question_ids=ids,
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=metadata,
                   output_path_base=output_file_base,
                   num_processes=num_processes,
                   metrics=metrics)
    if use_cache:
        norm_rules.save_cache(os.path.splitext(dataset_file)[0]+"__cache")
    return res


def validate_args(args):
    # input question(s) for decomposition are provided.
    assert args.preds_file and args.dataset_file

    # input files exist.
    if args.dataset_file:
        assert os.path.exists(args.dataset_file)
    if args.preds_file:
        assert os.path.exists(args.preds_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate QDMR predictions"
                                                 "Example: "
                                                 "--dataset_file=old_data_dev_low_level.csv --preds_file=old_data_dev_low_level_preds.csv --no_cache --output_file_base=old_data_full")
    parser.add_argument('--dataset_file', type=str, help='path to dataset file')
    parser.add_argument('--preds_file', type=str, help='path to a csv predictions file, with "decomposition" column')
    parser.add_argument('--allennlp', action='store_true', help='treat preds_file as a json predictions file generated by allennlp')
    parser.add_argument('--random_n', type=int, default=0,
                        help='choose n random examples from input file')
    parser.add_argument('--no_cache', action='store_true',
                        help="don't cache dependency parsing on normalized metrics")
    parser.add_argument('--num_processes', type=int, default=5,
                        help='number of processes for multiprocessing evaluation')
    parser.add_argument('--output_file_base', type=str, default=None, help='path to output file')
    parser.add_argument('--metrics', nargs='+', help='metrics to eval',
                        choices=['exact_match', 'match', 'structural_match', 'sari',
                                 'ged', 'structural_ged', 'ged_plus',
                                 'normalized_exact_match', 'normalized_match', 'normalized_structural_match', 'normalized_sari',
                                 'logical_form_em'],
                        default=['exact_match', 'match', 'structural_match', 'sari',
                                 'normalized_exact_match', 'normalized_match', 'normalized_structural_match', 'normalized_sari',
                                 'logical_form_em'])

    args = parser.parse_args()

    validate_args(args)
    res = evaluate_predictions(
        dataset_file=args.dataset_file,
        preds_file=args.preds_file,
        output_file_base=args.output_file_base,
        allennlp=args.allennlp,
        random_n=args.random_n,
        num_processes=args.num_processes,
        use_cache=not args.no_cache,
        metrics=args.metrics and [EvalMatrics(x) for x in args.metrics]
    )