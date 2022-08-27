import argparse
import json
import traceback
import logging

from datetime import datetime
import pandas as pd
import os
import pickle
import re

import spacy


def get_example_split_set_from_id(question_id):
    return question_id.split('_')[1]


def preprocess_seq2seq(examples: pd.DataFrame, lexicon_file: str, output_file_base:str) -> (pd.DataFrame, str):
    if output_file_base.startswith('test'):
        examples = examples.drop('decomposition')
    else:
        examples['decomposition'] = examples['decomposition'].apply(lambda x: process_target(x))
    examples['lexicon_tokens'] = examples['lexicon_tokens'].apply(lambda x: fix_references(x))
    examples.to_csv(f'{output_file_base}_seq2seq.csv', index=False)


def fix_references(string):
    return re.sub(r'#([1-9]+)', '@@\g<1>@@', string)


def process_target(target):
    if not (target and isinstance(target, str)):
        return None

    # replace multiple whitespaces with a single whitespace.
    target_new = ' '.join(target.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = target_new.split(';')
    new_parts = [re.sub(r'return', '', part.strip()) for part in parts]
    target_new = ' @@SEP@@ '.join([part.strip() for part in new_parts])

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    target_new = fix_references(target_new)

    return target_new.strip()


def sample_examples(examples: pd.DataFrame, configuration, is_strict:bool):
    df = examples

    logger.info("dataset distribution before sampling:")
    logger.info(df.groupby("dataset").agg("count"))
    for dataset in df.dataset.unique().tolist():
        if dataset in configuration or is_strict:
            drop_frac = 1 - configuration.get(dataset, 0)
            df = df.drop(df[df.dataset == dataset].sample(frac=drop_frac).index)

    logger.info("dataset distribution after sampling:")
    logger.info(df.groupby("dataset").agg("count"))

    return df


def parse_examples(examples: pd.DataFrame):
    parser = spacy.load('en_core_web_sm', disable=['ner'])
    parsed = {}
    for _, x in examples.iterrows():
        source_parsed = parser(x['question_text'])
        parsed[x['question_id']] = {'question_parsed':source_parsed}
    return parsed


_parsed_dict={}
def get_parsed(question_id: str, cache_dir:str = 'datasets/Break/QDMR', set_:str =None):
    try:
        set = set_ or question_id.split('_')[1]
        path = os.path.join(cache_dir, f'{set}_parsed.pkl')
        if path not in _parsed_dict:
            with open(path, 'rb') as f:
                parsed = pickle.load(f)
            _parsed_dict[path] = parsed
        parsed = _parsed_dict[path]
        return parsed[question_id]['question_parsed']
    except Exception as ex:
        raise ValueError(f'Could not load parsed cache for question {question_id} from {path}') from ex


def main(input_file, lexicon_file, output_dir,
         sample, sample_strict, is_to_parse):
    os.makedirs(output_dir, exist_ok=True)
    path_pref = os.path.join(output_dir,
                            f'{os.path.splitext(os.path.basename(input_file))[0]}')

    examples = pd.read_csv(input_file)
    assert all(x in examples for x in ['question_id', 'question_text', 'decomposition']), "missing columns"

    if 'dataset' not in examples:
        examples['dataset'] = examples['question_id'].apply(lambda x: x.split('_')[0])
        examples.to_csv(input_file, index=False)

    examples['lexicon_tokens'] = [
        json.loads(line)['allowed_tokens']
        for line in open(lexicon_file, "r").readlines()
    ]

    if sample:
        assert output_dir != os.path.dirname(input_file)
        examples = sample_examples(examples, sample, is_strict=sample_strict)
        examples.to_csv(f'{path_pref}.csv', columns=[x for x in examples.columns if x not in ['lexicon_tokens']], index=False)
        examples[['question_text', 'lexicon_tokens']]\
            .rename(columns={'question_text': 'source', 'lexicon_tokens': 'allowed_tokens'})\
            .to_json(os.path.join(output_dir, os.path.basename(lexicon_file)), orient='records', lines=True)
        logger.info(f"left with {len(examples)} examples after sampling.")

    if is_to_parse:
        parsed_examples = parse_examples(examples)
        with open(f"{path_pref}_parsed.pkl", 'wb') as fd:
            pickle.dump(parsed_examples, fd)

    preprocess_seq2seq(examples.copy(), lexicon_file, path_pref)

    logger.info("done!\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""preprocess examples
                    example:
                    python scripts/data_processing/preprocess_examples.py datasets/Break/QDMR --output_dir cwq-only
                    --sample '{"CWQ": 1.0}' --sample_strict
                    """
    )
    parser.add_argument('input_dir', type=str, help='path to input dir (assumes train/dev/test.csv')
    parser.add_argument('-o', '--output_dir', type=str, default="",
                        help='path to output directory (inside the input_dir): <input_dir>/<output_dir> (Default: None)')
    parser.add_argument('--sample', type=json.loads, default="{}",
                        help='json-formatted string with dataset down-sampling configuration, '
                             'for example: {"ATIS": 0.5, "CLEVR": 0.2}. By default, the rest datasets are fully taken')
    parser.add_argument('--sample_strict', action='store_true',
                        help='drop any dataset that is not in sample config')
    parser.add_argument('--parse', action="store_true", help='whether to process the examples with spaCy')
    args = parser.parse_args()
    assert os.path.exists(args.input_dir)
    assert args.output_dir or not args.sample

    output_dir = os.path.join(args.input_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(format='%(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(filename=os.path.join(output_dir, 'data_process.log'),
                             mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(f'========================= {datetime.now().strftime("%d.%m.%Y %H:%M:%S")} ==========================')

    for basename in ['dev', 'train', 'test']:
        try:
            logger.info(f'----------- {basename.upper()} -------------:')
            input_file = os.path.join(args.input_dir, f'{basename}.csv')
            lexicon_file = os.path.join(args.input_dir, f'{basename}_lexicon_tokens.json')
            main(input_file=input_file, lexicon_file=lexicon_file, output_dir=output_dir,
                 sample=args.sample, sample_strict=args.sample_strict,
                 is_to_parse=args.parse)
        except Exception as ex:
            logger.exception(f'exception on {basename}: {str(ex)}')
            traceback.print_exc()

