import pandas as pd
from ast import literal_eval
import os
import json
from datetime import date

from qdecomp_with_dependency_graphs.dependencies_graph.config.configuration_loader import config, save
from qdecomp_with_dependency_graphs.dependencies_graph.extractors import BaseTokensDependenciesExtractor
from qdecomp_with_dependency_graphs.scripts.data_processing.preprocess_examples import get_parsed


def get_tokens_dependencies_generator(dataset: pd.DataFrame, tokens_dependencies_extractor: BaseTokensDependenciesExtractor,
                                      error_handler=None):
    for index, row in dataset.iterrows():
        question_id, question_text, decomposition, operators, *_ = row
        operators = [x.strip() for x in literal_eval(operators)]

        try:
            tok_dep = tokens_dependencies_extractor.extract(question_id, question_text, decomposition, operators)
            yield question_id, tok_dep
        except Exception as ex:
            if error_handler:
                error_handler(question_id=question_id, exception=ex)
                yield question_id, None
            else:
                raise ex

EXTRA_TOKEN_TAG = "NONE"
def get_extra_tokens():
    extra_tokens = config.tokens_dependencies_extractor.get_extra_tokens()
    pos_tags = [EXTRA_TOKEN_TAG] * len(extra_tokens)
    return extra_tokens, pos_tags


def main(
    dataset_file: str,
    dest_dir: str,
    tokens_dependencies_extractor: BaseTokensDependenciesExtractor):

    os.makedirs(dest_dir, exist_ok=True)

    # save config
    save(dest_dir)

    df = pd.read_csv(dataset_file)

    tokens_dependencies = get_tokens_dependencies_generator(
        dataset=df,
        tokens_dependencies_extractor=tokens_dependencies_extractor,
        error_handler=lambda question_id, exception: print(f'error in {question_id}: {str(exception)}'))

    output = []
    total = 0
    for question_id, tok_dep in tokens_dependencies:
        total+=1
        try:
            if tok_dep is None:
                continue
            parsed_question = get_parsed(question_id, cache_dir=os.path.dirname(dataset_file))
            tokens = tok_dep.tokens()
            parsed_tokens = [t for t in parsed_question if t.text]

            output.append({
                'tokens': [{'text': x.text, 'bio':x.tag, 'lemma':y.lemma_, 'pos': y.pos_, 'tag': y.tag_} for x,y in zip(tokens, parsed_tokens)],
                'extra_tokens': [{'text': x.text, 'pos': EXTRA_TOKEN_TAG, 'tag': EXTRA_TOKEN_TAG} for x in list(tokens)[len(parsed_tokens):]],
                'deps': [(u,v,str(dep.dep_type)) for u,v,dep in tok_dep.dependencies()],
                'metadata': {'question_id': question_id}
            })
        except Exception as ex:
            print(f'error in {question_id}: {str(ex)}')

    with open(os.path.join(dest_dir, f'{os.path.basename(dataset_file).split(".")[0]}_dependencies_graph.json'), 'wt') as f:
        f.writelines([json.dumps(x)+'\n' for x in output])
    with open(os.path.join(dest_dir, f'{os.path.basename(dataset_file).split(".")[0]}_dependencies_graph_summary.txt'), 'wt') as f:
        f.write(f'produced samples: {len(output)}/{total} ({len(output)*100/total:.3f}%)')
    print(f'produced samples: {len(output)}/{total} ({len(output)*100/total:.3f}%)')


def prepare_questions_only(
        dataset_file: str,
        dest_dir: str,
        ):

    os.makedirs(dest_dir, exist_ok=True)

    # save config
    save(dest_dir)

    df = pd.read_csv(dataset_file)

    output = []
    total = 0
    def fix_tag(tag: str):
        return {
            'NFP': 'NONE',
        }.get(tag, tag)

    set_ = os.path.splitext(os.path.basename(dataset_file))[0]
    for _, row in df.iterrows():
        total+=1
        question_id = row['question_id']
        try:
            parsed_question = get_parsed(question_id, cache_dir=os.path.dirname(dataset_file), set_=set_)
            tokens = parsed_question
            extra_tokens, extra_pos = get_extra_tokens()

            output.append({
                'tokens': [{'text': x.text, 'bio': None, 'lemma':x.lemma_, 'pos': x.pos_, 'tag': fix_tag(x.tag_)} for x in tokens],
                'extra_tokens': [{'text': x, 'pos': EXTRA_TOKEN_TAG, 'tag': EXTRA_TOKEN_TAG} for x in extra_tokens],
                'metadata': {'question_id': question_id}
            })
        except Exception as ex:
            print(f'error in {question_id}: {str(ex)}')

    base_name = f'{set_}_dependencies_graph__questions_only'
    with open(os.path.join(dest_dir, f'{base_name}.json'), 'wt') as f:
        f.writelines([json.dumps(x)+'\n' for x in output])
    with open(os.path.join(dest_dir, f'{base_name}_summary.txt'), 'wt') as f:
        f.write(f'produced samples: {len(output)}/{total} ({len(output)*100/total:.3f}%)')
    print(f'produced samples: {len(output)}/{total} ({len(output)*100/total:.3f}%)')


def replaced_gold_bio_by_learned(dataset_file:str, dep_graphs_file: str, learned_bio_file: str):
    df = pd.read_csv(dataset_file)
    with open(dep_graphs_file, 'rt') as f:
        dep_graphs = [json.loads(x.strip()) for x in f.readlines()]
    with open(learned_bio_file, 'rt') as f:
        learned_bio = [json.loads(x.strip()) for x in f.readlines()]
    df['bio'] = learned_bio
    for x in dep_graphs:
        question_id = x['metadata']['question_id']
        y = df[df['question_id']==question_id]['bio'].iloc[0]
        x['tokens'] = [{**t, 'bio': b} for t,b in zip(x['tokens'],y['tags'])]
    with open(os.path.splitext(dep_graphs_file)[0]+'_learned_bio_crf.json', 'wt') as f:
        f.writelines(json.dumps(x)+'\n' for x in dep_graphs)


if __name__ == '__main__':
    for dataset in ["dev", "train"]:
        main(
            dataset_file=f'datasets/Break/QDMR/{dataset}.csv',
            dest_dir=f'datasets/Break/QDMR',
            tokens_dependencies_extractor=config.tokens_dependencies_extractor
                #spans_file=f'datasets/Break/QDMR/{dataset}_spans.json'),
        )

    for set_ in ['dev', 'test']:
        prepare_questions_only(
            dataset_file=f'datasets/Break/QDMR/{set_}.csv',
            dest_dir=f'datasets/Break/QDMR',
        )