from abc import ABC

import argparse
from overrides import overrides
import re
import os
import json
from collections import defaultdict

import pandas as pd
from typing import List, Dict, Callable, Optional

from pathlib import Path

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.evaluate_dep_graph import get_logical_form_tokens_formatters, get_samples, evaluate
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher, QDMRStepTokensDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.qdmr_to_logical_form_tokens import QDMRProgramToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.scripts.data_processing.preprocess_examples import fix_references
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import get_predictions_from_allennlp_preds_file
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import QDMRProgramBuilder, QDMRStep, ArgumentType, QDMROperator

import logging

logger = logging.getLogger(__name__)

##################################
#       Creation                 #
##################################

def fix_args(arg_val: str):
    return fix_references(arg_val).replace('#REF', '@@REF@@')

def unfix_args(arg_val: str):
    return re.sub(r'@@([1-9]+)@@', '#\g<1>', arg_val).replace('@@REF@@', '#REF')


class Processor(ABC):
    def __init__(self):
        self.name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()

    def to_string(self, program: List[QDMRStep]) -> str:
        raise NotImplementedError()

    def to_program(self, str_rep: str)->List[QDMRStep]:
        raise NotImplementedError()

    @staticmethod
    def get_by_name(name: str) -> 'Processor':
        return {x.name: x for x in Processor.get_processors()}[name]

    @staticmethod
    def get_processors() -> List['Processor']:
        return [SpecialTokens(), SectorsRep()]


class SectorsRep(Processor):
    SEP = '@@SEP@@'
    PROP = '@@PROP@@'
    ARG_NAME = '@@ARG@@'
    ARG_VALUE = '@@ARG_VAL@@'

    @overrides
    def to_string(self, program: List[QDMRStep]) -> str:
        def step_to_str(i:int, step: QDMRStep):
            operator = str(step.operator)
            properties = step.properties and ' '.join([self.PROP]+list(sorted(step.properties)))
            args = ' '.join(sorted(f"{self.ARG_NAME} {k.replace('-', '_')} {self.ARG_VALUE} {fix_args(v_item)}" for k, v in step.arguments.items() for v_item in v))
            return ' '.join(x for x in [operator, properties, args] if x)
        return f" {self.SEP} ".join(step_to_str(i+1, step) for i, step in enumerate(program))


    @overrides
    def to_program(self, str_rep: str)->List[QDMRStep]:
        str_rep = unfix_args(str_rep)
        steps_str = re.split(rf'\s+{self.SEP}\s+', str_rep)
        steps = []
        for step_str in steps_str:
            parts = re.split(f'({self.PROP}|{self.ARG_NAME})', step_str)
            operator = QDMROperator(parts[0].strip())
            parts = parts[1:]
            props = []
            args: Dict[ArgumentType, List[str]] = defaultdict(list)
            for i in range(0, len(parts), 2):
                val = parts[i+1].strip()
                if parts[i] == self.PROP:
                    props.extend(re.split(r'\s+', val))
                elif parts[i] == self.ARG_NAME:
                    arg_parts = re.split(f'\s+{self.ARG_VALUE}\s+', val)
                    args[ArgumentType(arg_parts[0].replace('_', '-'))].extend(arg_parts[1:])
            assert all(x.get_operator() == operator for x in args.keys()), f'mismatch operators {operator}, {args.keys()}'
            props = list(set(props))
            steps.append(QDMRStep(None, operator=operator, properties=props, arguments=args))
        return steps


class SpecialTokens(Processor):
    SEP = '@@SEP@@'

    @overrides
    def to_string(self, program: List[QDMRStep]) -> str:
        def step_to_str(i:int, step: QDMRStep):
            properties = step.properties and f'@@{step.operator}_prop@@ '+' '.join(sorted(step.properties))
            args = ' '.join(sorted(f"@@{k.replace('-','_')}@@ {fix_args(v_item)}" for k, v in step.arguments.items() for v_item in v))
            return ' '.join(x for x in [properties, args] if x)
        return f" {self.SEP} ".join(step_to_str(i+1, step) for i, step in enumerate(program))


    @overrides
    def to_program(self, str_rep: str)->List[QDMRStep]:
        str_rep = unfix_args(str_rep)
        steps_str = re.split(rf'\s+{self.SEP}\s+', str_rep)
        steps = []
        for step_str in steps_str:
            parts = re.split('(@@\w+_\w+@@)', step_str)
            assert parts[0] == '', f'invalid format {step_str}'
            parts = parts[1:]
            props = defaultdict(list)
            args: Dict[ArgumentType, List[str]] = defaultdict(list)
            for i in range(0, len(parts), 2):
                separator = parts[i].replace('@', '').replace('_', '-')
                val = parts[i+1].strip()
                if separator.endswith('-prop'):
                    props[QDMROperator(separator.split('-')[0])].extend(re.split(r'\s+', val))
                else:
                    args[ArgumentType(separator)].append(val)
            operator_candidates = set(list(props.keys())+[x.get_operator() for x in args.keys()])
            assert len(operator_candidates) == 1, f'mismatch operators {operator_candidates}'
            operator = operator_candidates.pop()
            props = list(set([x for v in props.values() for x in v]))
            steps.append(QDMRStep(None, operator=operator, properties=props, arguments=args))
        return steps


def create(formatter: Processor):
    def qdmr_to_lf(decomp: str):
        try:
            builder = QDMRProgramBuilder(decomp)
            builder.build()
            return formatter.to_string(builder.steps)
        except:
            return "ERROR"

    for dataset in ['datasets/Break/QDMR/dev.csv', 'datasets/Break/QDMR/train.csv']:
        dest_path_base = os.path.splitext(dataset)[0]+'_lf_'+ formatter.name
        df = pd.read_csv(dataset)
        # df = df.sample(n=100)
        df['original_decomposition'] = df['decomposition']
        df['decomposition'] = df['decomposition'].apply(lambda x: qdmr_to_lf(x))
        stat = {'total': len(df)}
        df = df[df['decomposition'] != 'ERROR']
        stat['no_error'] = len(df)
        stat['no_error_rate'] = stat['no_error']/stat['total']

        df.to_csv(dest_path_base+".csv", index=False)
        with open(dest_path_base+'__summary.json', 'wt') as fp:
            json.dump(stat, fp)


def debug_processor(samples: List[str], processor: Processor, silent:bool = False):
    success = 0
    for x in samples:
        try:
            x_prog = processor.to_program(x)
            x_str = processor.to_string(x_prog)
            success += int(x == x_str)
            if not silent:
                print(f'{"SUCCESS" if x==x_str else "FAIL"}: {x} => {x_str}')
        except Exception as ex:
            print(f'ERROR: {str(ex)}')
    print(f'success: {success}/{len(samples)} ({success/len(samples)*100}%)')


##################################
#       Eval                     #
##################################

def get_logical_form_tokens_formatter(processor: Processor) -> Callable[[str, str, str, Optional[dict]], str]:
    program_converter = QDMRProgramToQDMRStepTokensConverter()
    formatter = LogicalFromStructuralMatcher()

    def _add_meta(meta: dict, qdmr_graph: QDMRStepTokensDependencies, reorder: bool):
        if meta is not None:
            try:
                # meta['original'] = formatter.graph_key(qdmr_graph)
                meta['original'] = qdmr_graph.to_string(reorder=reorder)
            except Exception as ex:
                meta['original'] = f'ERROR: {str(ex)}'

    def lf_formatter(question_id: str, question_text: str, decomposition: str, meta: dict = None) -> str:
        try:
            program = processor.to_program(decomposition)
            qdmr_graph: QDMRStepTokensDependencies = program_converter.convert(question_id=question_id, question_text=question_text, program=program)
            _add_meta(meta, qdmr_graph, False)
            formatter.normalize_logical_graph(question_id, question_text, qdmr_graph)
            return formatter.graph_key(qdmr_graph)
        except Exception as ex:
            logging.exception(f'{question_id}: {str(ex)}')
            return "ERROR"
    return lf_formatter


def evaluate_lf(dataset: pd.DataFrame, dest_path: str, preds_file:str):
    def get_predictions():
        nonlocal dataset
        predictions, question_ids = get_predictions_from_allennlp_preds_file(preds_file, format=False)
        question_ids_to_predictions = {k: v for k, v in zip(question_ids, predictions)}
        dataset = dataset[dataset['question_id'].isin(question_ids)]
        # assert len(dataset) == len(question_ids)
        for _, row in dataset.iterrows():
            yield question_ids_to_predictions[row['question_id']]

    qdmr_formatter, _ = get_logical_form_tokens_formatters()
    processor = [x for x in Processor.get_processors() if x.name in preds_file][0]
    lf_formatter = get_logical_form_tokens_formatter(processor)
    res = evaluate(samples=get_samples(dataset),
                   pred_decomps=get_predictions(),
                   formatters_dict={
                       'logical_form_tokens': (qdmr_formatter, lf_formatter)
                   },
                   dest_path=dest_path
                   )
    return res


if __name__ == '__main__':
    def run_create(args):
        for formatter in [SpecialTokens(), SectorsRep()]:
            try:
                create(formatter)
            except Exception as ex:
                logger.exception(f'failed to create using {formatter}')

    def run_debug_create(args):
        processor = Processor.get_by_name(args.processor)
        samples = pd.read_csv(f'datasets/Break/QDMR/dev_lf_{processor.name}.csv').sample(args.n)
        debug_processor(samples['decomposition'].to_list(), processor, args.silent)

    def run_evaluate(args):
        assert args.preds_file or (args.root and args.pattern)
        if args.root:
            preds_files = [str(x) for x in Path(args.root).rglob(args.pattern)]
            if not preds_files:
                logger.warning(f"No predictions file found in {args.root} (pattern {args.pattern})")
        else:
            preds_files = [args.preds_file]

        for preds_file in preds_files:
            try:
                dest_path = os.path.splitext(preds_file)[0]+'__lf_eval.csv'
                if args.dest_dir:
                    dest_path = os.path.join(args.dest_dir, os.path.basename(dest_path))
                evaluate_lf(dataset=df,
                            dest_path=dest_path,
                            preds_file=preds_file)
            except Exception as ex:
                logger.exception(f'failed to evaluate: preds_file {preds_file}')

    parser = argparse.ArgumentParser(description='logical form dataset')
    subparser = parser.add_subparsers()

    create_parser = subparser.add_parser('create', help='create LF dataset')
    create_parser.set_defaults(func=run_create)

    debug_parser = subparser.add_parser('debug', help='debug LF conversion')
    debug_parser.set_defaults(func=run_debug_create)
    debug_parser.add_argument('-p', '--processor', choices=[x.name for x in Processor.get_processors()], help='proccesor')
    debug_parser.add_argument('-n', type=int, default=10, help='samples number')
    debug_parser.add_argument('--silent', action='store_true', help='skip plots')

    eval_parser = subparser.add_parser('eval', help='evaluate predictions')
    eval_parser.set_defaults(func=run_evaluate)
    eval_parser.add_argument('-p', '--preds_file', type=str, required=False, help='predictions file')
    eval_parser.add_argument('-r', '--root', type=str, required=False, help='predictions root dir')
    eval_parser.add_argument('--pattern', type=str, required=False, help='pattern for predictions files (when root is given)')
    eval_parser.add_argument('-d', '--dest-dir', required=False, type=str,
                             help='destination directory (default: same as evaluation file)')

    df = pd.read_csv('datasets/Break/QDMR/dev.csv')
    args = parser.parse_args()
    args.func(args)