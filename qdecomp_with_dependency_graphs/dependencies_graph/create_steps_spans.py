import argparse
import json
import os
import random
from ast import literal_eval
from dataclasses import dataclass
from typing import List

import pandas as pd

from qdecomp_with_dependency_graphs.dependencies_graph.config.configuration_loader import config, save
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import StepsSpans
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors import BaseSpansExtractor
from qdecomp_with_dependency_graphs.utils.data_structures import EnhancedJSONEncoder

import logging

_logger = logging.getLogger(__name__)


##############################################
#       Create data files                    #
##############################################

def create_spans(spans_extractor: BaseSpansExtractor, dataset_file: str, output_file: str):
    assert os.path.exists(dataset_file), f'dataset {dataset_file} does not exist'
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save(os.path.dirname(output_file))

    df = pd.read_csv(dataset_file)
    spans = []
    for index, row in df.iterrows():
        question_id, question_text, decomposition, operators, *_ = row
        operators = [x.strip() for x in literal_eval(operators)]

        try:
            steps_spans = spans_extractor.extract(question_id=question_id, question=question_text, decomposition=decomposition, operators=operators)
            spans.append({
                'steps_spans': steps_spans.to_dict(),
                'metadata': {'question_id': question_id}
            })
        except Exception as ex:
            _logger.exception(ex)

    with open(output_file, 'wt') as f:
        f.writelines([json.dumps(x, cls=EnhancedJSONEncoder)+'\n' for x in spans])


def create_tags(spans_file: str, output_file: str):
    assert os.path.exists(spans_file), f'spans file {spans_file} does not exist'
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = []
    with open(spans_file, 'rt') as rf:
        for row in rf.readlines():
            content = json.loads(row.strip())
            steps_spans: StepsSpans = StepsSpans.from_dict(content['steps_spans'])
            lines.append('\t'.join([f'{token.text}###{token.bio_tag}' for token in steps_spans.tokens()]))
    with open(output_file, 'wt') as wf:
        wf.writelines(x+'\n' for x in lines)


##############################################
#       Debug and Tests                      #
##############################################

def print_alignments_from_dataset(spans_extractor: BaseSpansExtractor, dataset_file:str,
                                  random_n:int=None, question_ids:[str]=None):
    df = pd.read_csv(dataset_file)

    if question_ids:
        df = df[df['question_id'].isin(question_ids)]
    else:
        indexes = random.sample(range(len(df.index)), k=random_n)
        df = df.iloc[indexes]

    for index, row in df.iterrows():
        question_id, question_text, decomposition, operators, *_ = row
        operators = [x.strip() for x in literal_eval(operators)]
        steps_spans = spans_extractor.extract(question_id=question_id, question=question_text, decomposition=decomposition, operators=operators)
        print(question_id)
        print(steps_spans.get_alignments_str())


def run_tests(spans_extractor: BaseSpansExtractor, verbose: bool = True):
    @dataclass
    class TestCase:
        context: str
        question: str
        decomposition: str
        operators: List[str]
        desired_spans: List[List[str]]

    tests = [
        TestCase(
            context="43283, SPIDER_train_6359, prefer sequential, take refs",
            question="Show the school name and driver name for all school buses.",
            decomposition="return school buses ;return schools of  #1 ;return names of  #2 ;return drivers of  #1 ;return names of  #4 ;return #3 ,  #5",
            operators=['select', 'project', 'project', 'project', 'project', 'union'],
            desired_spans=[['school buses'], ['school'], ['name'], ['driver'], ['name'], ['and']]
        ),
        TestCase(
            context="6321,  CLEVR_train_13444, align the rest to the last step?",
            question="Is the blue cylinder the only metal object present?",
            decomposition="return blue cylinder ;return metal objects ;return number of  #1 ;return number of  #2 ;return is  #3 the  same as #4",
            operators=['select', 'select', 'aggregate', 'aggregate', 'boolean'],
            desired_spans=[['blue cylinder'], ['metal object'], [], [], ['Is', 'only']]
        ),
        TestCase(
            context="4245, CLEVR_train_10007, multiply vs multiplication (use lexical)",
            question="what number is the cyan objects multiply by blue objects?",
            decomposition="return cyan objects ;return blue objects ;return number of  #1 ;return number of  #2 ;return multiplication of #3 and  #4",
            operators=['select', 'select', 'aggregate', 'aggregate', 'arithmetic'],
            desired_spans=[['cyan objects'], ['blue objects'], [], [], ['multiply']]
        ),
        TestCase(
            context="40079, SPIDER_train_3456, multipass alignment",
            question="Give the country id and corresponding count of cities in each country.",
            decomposition="return countries ;return country  ids of  #1 ;return cities of  #1 ;return number of #3 for each #1 ;return #2 ,   #4",
            operators=['select', 'project', 'project', 'group', 'union'],
            desired_spans=[['country'], ['country id'], ['cities'], ['each'], ['and']]
        ),
        TestCase(
            context="43253, SPIDER_train_6331, names of (X and Y)",
            question="What are the names of all cities and states?",
            decomposition="return cities ;return names of #1 ;return states ;return names of #3 ;return #2 ,  #4",
            operators=['select', 'project', 'select', 'project', 'union'],
            desired_spans=[['cities'], ['names'], ['states'], ['name'], ['and']]
        ),
        TestCase(
            context="23811, DROP_train_history_743_e5b52630-cee5-4e66-8465-91341d7e060a, omit #",
            question="How many years did it take the proportion of Southern Irish Protestants to decline from 10 to 3%?",
            decomposition="return Southern Irish Protestants ;return the  proportion of #1 in  % ;return year when  #2 was  10 % ;return year when  #2 was  3 % ;return the  difference of #4 and  #3",
            operators=['select', 'project', 'project', 'project', 'arithmetic'],
            desired_spans=[['Southern Irish Protestants'], ['the proportion of'], ['years','10'], ['years', '3'], ['How many', 'to decline']]
        ),
        TestCase(
            context="35518, NLVR2_train_train-7752-3-1, left vs leave - prefer exact match",
            question="If all of the gorillas are holding food in their left hand.",
            decomposition="return gorillas ;return food ;return hand of #1 ;return #3 that is left ;return #1 holding #2 in  #4 ;return the  number of  #1 ;return the  number of  #5 ;return if  #6 is equal to  #7",
            operators=['select', 'select', 'project', 'filter', 'filter', 'aggregate', 'aggregate', 'boolean'],
            desired_spans=[['gorillas'], ['food'], ['hand'], ['left'], ['holding'], [], [], ['If all']]
        ),
        # TestCase(
        #     context="",
        #     question="",
        #     decomposition="",
        #     desired_spans=None
        # ),
    ]

    def is_valid(test: TestCase, steps_spans: StepsSpans):
        if not test.desired_spans:
            return True
        for (desire, (_, aligned)) in zip(test.desired_spans, steps_spans.step_spans_text()):
            if len(desire)>len(aligned):
                return False
            for d in desire:
                if d not in aligned:
                    return False
        return True

    invalids = 0
    for test in tests:
        steps_spans, _ = spans_extractor.extract(question_id='', question=test.question, decomposition=test.decomposition, operators=test.operators)
        valid = is_valid(test, steps_spans)
        if not valid:
            invalids += 1
        if verbose or not valid:
            steps_context = test.desired_spans
            print(test.context, steps_spans.get_alignments_str(steps_context))
    print(f'{invalids}/{len(tests)} invalid tests')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    spans_extractor = config.spans_extractor

    def sample_command(args):
        assert args.data
        assert args.random_n or args.question_ids
        print_alignments_from_dataset(spans_extractor=spans_extractor, dataset_file=args.data, random_n=args.random_n,
                                      question_ids=args.question_ids)


    def test_command(args):
        run_tests(spans_extractor=spans_extractor, verbose=args.verbose)


    def extract_command(args):
        assert args.data
        output_file = args.output_file or os.path.splitext(args.data)[0] + '_spans.json'
        create_spans(spans_extractor=spans_extractor, dataset_file=args.data, output_file=output_file)


    def create_tags_command(args):
        assert args.input_file
        output_file = os.path.splitext(args.input_file)[0] + '_BIO.txt'
        create_tags(spans_file=args.input_file, output_file=output_file)


    parser = argparse.ArgumentParser(description='Extract spans for a QDMR decomposition steps')
    subparsers = parser.add_subparsers()

    # sample
    parser_sample = subparsers.add_parser('sample', help='align samples from given dataset')
    parser_sample.set_defaults(func=sample_command)
    parser_sample.add_argument('--data', type=str, default=r'datasets/Break/QDMR/train.csv',
                               help='csv dataset file (default: datasets/Break/QDMR/train.csv)')
    parser_sample.add_argument('-n', '--random_n', type=int, help='amount of random samples')
    parser_sample.add_argument('-q', '--question_ids', nargs='+', type=str,
                               help='specific question_ids')

    # test
    parser_test = subparsers.add_parser('test', help='run tests')
    parser_test.set_defaults(func=test_command)
    parser_test.add_argument('--verbose', action='store_true', help='verbose mode')

    # extract
    parser_extract = subparsers.add_parser('extract', help='extract spans from given dataset')
    parser_extract.set_defaults(func=extract_command)
    parser_extract.add_argument('--data', type=str, default=r'datasets/Break/QDMR/train.csv',
                                help='csv dataset file (default: datasets/Break/QDMR/train.csv)')
    parser_extract.add_argument('-o', '--output_file', type=str, required=False,
                                help='destination json file (default: <data-dir>/<data-set>_spans.json)')

    # create tags
    parser_tags = subparsers.add_parser('tags', help='create sequence tags format for training')
    parser_tags.set_defaults(func=create_tags_command)
    parser_tags.add_argument('-i', '--input_file', type=str,
                             help='spans file ("extract" command output) to convert to BIO-tags')

    args = parser.parse_args()
    args.func(args)