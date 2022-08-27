from typing import Tuple
import argparse

import os
from ast import literal_eval
import re
import statistics
import pandas as pd
import spacy
from spacy.attrs import ORTH
from spacy.tokens import Doc


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


###########################################
#               Statistics                #
###########################################

def decomposition_length_statistics(data_df: pd.DataFrame):
    # average number of steps per dataset
    _print_title('Avg Number of Steps')
    df = data_df[["dataset", "num_steps"]].groupby(["dataset"]).agg("mean")
    print(df.round(decimals=3))


#   OPERATORS   ###########################################

def operators_statistics(data_df: pd.DataFrame):
    total_operators_count = _count_operators(data_df)

    # operators histogram - per dataset and per num_steps
    _print_title('Operator Histogram')
    for agg_field in ["dataset", "num_steps"]:
        df = data_df[[agg_field]+list(sorted(total_operators_count.keys()))].groupby(agg_field).agg("mean")
        print(df.round(decimals=3))

    # operation histogram (for num_steps) per dataset
    _print_title('Operator Histogram per Dataset')
    for dataset in data_df['dataset'].unique():
        print(f'Dataset {dataset}:')
        ds_metadata = data_df[data_df['dataset'] == dataset]
        df = ds_metadata[list(sorted(total_operators_count.keys()))].astype(bool)
        df["num_steps"] = ds_metadata["num_steps"]
        df = df.groupby(["num_steps"]).agg("sum")
        print(df.round(decimals=3))


def _count_operators(metadata):
    total_operators_count = {}
    for i, x in metadata["operators"].iteritems():
        operators_count = {}
        ops = [y.strip() for y in x.replace("[", "").replace("]", "").replace("'", "").split(',')]
        for o in ops:
            prev = operators_count.get(o, 0)
            operators_count[o] = prev + 1
        for o in operators_count:
            if o in total_operators_count:
                total_operators_count[o].append(operators_count[o])
            else:
                counts = [0] * i
                total_operators_count[o] = counts + [operators_count[o]]
        for o in total_operators_count:
            if o not in operators_count:
                total_operators_count[o].append(0)
    for o in total_operators_count:
        metadata[o] = total_operators_count[o]
    return total_operators_count


#   VOCABULARY   ###########################################

def vocabulary_variance(data_df: pd.DataFrame):
    parser = spacy.load('en_core_web_sm')

    def count_per_dataset(count_by_attr, exclude):
        stats = []
        for dataset in data_df['dataset'].unique():
            df = data_df[data_df['dataset']==dataset]
            corpus = ' '.join(df['question_text'])
            doc: Doc = parser(corpus)
            counts = doc.count_by(count_by_attr, exclude=exclude)
            distinct_tokens = len(counts)
            num_of_tokens = sum(counts.values())
            total_tokens = len(doc)

            stats.append({
                'dataset': dataset,
                'distinct_tokens': f'{distinct_tokens}/{num_of_tokens}({distinct_tokens/num_of_tokens*100:.2f}%)',
                'median_appearance': statistics.median(counts.values()),
                'total': f'{num_of_tokens}/{total_tokens}({num_of_tokens/total_tokens*100:.2f}%)'
            })
        print(pd.DataFrame.from_records(stats))
    _print_title('Vocab - no punctuations')
    count_per_dataset(ORTH, lambda x: x.is_punct)
    _print_title('Vocab - no punctuations no stops')
    count_per_dataset(ORTH, lambda x: x.is_punct or x.is_stop)
    _print_title('Vocab - punctuations only')
    count_per_dataset(ORTH, lambda x: not x.is_punct)
    _print_title('Vocab - stops only')
    count_per_dataset(ORTH, lambda x: not x.is_stop)


#   PATTERNS   ###########################################

def check_pattern(data_df:pd.DataFrame, patterns: [Tuple[str,str]]):
    def is_match(row, pattern: str, operator:str):
        operators = literal_eval(row['operators'])
        steps = row['decomposition'].split(';')
        for i, x in enumerate(operators):
            if (operator == "*" or x == operator) and re.search(pattern, steps[i]):
                return True
        return False
    for operator, pattern in patterns:
        count = sum(1 if is_match(x, pattern, operator) else 0
                    for _, x in data_df.iterrows())
        print(f'{count}/{len(data_df)} ({count/len(data_df)*100:.2f}%)\t\tfor {operator}, {pattern}')


def refs_count_per_operator(data_df:pd.DataFrame):
    _print_title('number of references per operator')
    steps = data_df['decomposition'].apply(lambda x: x.split(';'))
    operators = data_df['operators'].apply(lambda x: literal_eval(x))
    data_df['step_operator_pair'] = [list(zip(x,y)) for x,y in zip(steps,operators)]
    df = data_df.explode('step_operator_pair')
    df['step'] = df['step_operator_pair'].apply(lambda x: x[0])
    df['operator'] = df['step_operator_pair'].apply(lambda x: x[1])
    df = df.drop('step_operator_pair', axis=1)
    df['refs'] = df['step'].apply(lambda x: len(re.findall(r'#\d+',x)))
    grouped = df.groupby(['operator'])['refs'].unique()
    for operator, refs in grouped.iteritems():
        print(f'{operator}: {sorted(refs)}')
    return df


#   UTILS   ###########################################

def run_on_dataset(data_dir: str, *funcs):
    for set in ['dev', 'train']:
        print()
        print(f'========================= {set.upper()} ============================')
        data_df = pd.read_csv(os.path.join(data_dir,f'{set}.csv'))
        data_df["num_steps"] = data_df["decomposition"].apply(lambda x: len(x.split(";")))

        for f in funcs:
            f(set, data_df)


def _print_title(title: str):
    print()
    print(title.upper())
    print('----------------------------')


def format_print_for_excel(text: str):
    """
    Format print text of aggregated pandas dataframe, so it can be copied to excel
    :param text: printed table
    :return: formatted print
    """
    print('\n'.join('\t'.join(y.strip() for y in x.split()) for x in text.split('\n')))


if __name__ == '__main__':
    # format_print_for_excel("""  """)
    def run_data_statistics(args):
        run_on_dataset(
            args.data_dir,
            lambda _, x: decomposition_length_statistics(x),
            lambda _, x: operators_statistics(x),
            lambda _, x: vocabulary_variance(x),
            lambda set, x: refs_count_per_operator(x).to_csv(f'_debug/{set}_refs_count.csv'),
        )

    def run_check_patterns(args):
        patterns = [(p_list[0], p) for p_list in args.patterns for p in p_list[1:]]
        run_on_dataset(
            args.data_dir,
            lambda _, x: check_pattern(x, patterns=patterns),
        )

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    stats_parser = subparser.add_parser('stats', help='run data statistics')
    stats_parser.set_defaults(func=run_data_statistics)
    stats_parser.add_argument('--data-dir', default="datasets/Break/QDMR",
                             help='Data directory. Default: "datasets/Break/QDMR/"')

    patterns_parser = subparser.add_parser('pattern', help='check frequency of patterns in the data')
    patterns_parser.set_defaults(func=run_check_patterns)
    patterns_parser.add_argument('--data-dir', default="datasets/Break/QDMR",
                              help='Data directory. Default: "datasets/Break/QDMR/"')
    patterns_parser.add_argument('-p', '--patterns', type=str, action='append', nargs='+',
                                 help='patterns to check. '
                                      'format: -p <operator> <regex> <regex> ... -p <operator> <regex> ...')

    args = parser.parse_args()
    args.func(args)
