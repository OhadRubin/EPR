import argparse
import traceback
from typing import Iterable

import pandas as pd
import random
from ast import literal_eval
import os
import sys
import shutil

import qdecomp_with_dependency_graphs.utils.html as utils_html
from qdecomp_with_dependency_graphs.dependencies_graph.config.configuration_loader import config, save
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.evaluate_dep_graph import evaluate_dep_graph
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import evaluate, evaluate_predictions, format_qdmr

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import StepsSpans, StepsDependencies, TokensDependencies, SpansDependencies


def to_dependencies_and_back_from_dataset(dataset_file:str,
                                          questions_ids: [int] = None, indexes:[int] = None, random_n: int = None,
                                          dest_dir: str = None,
                                          plot_dependencies: bool = False,
                                          is_eval: bool = True,
                                          func=None):

    plot_dir = None
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        save(dest_dir)
        if plot_dependencies:
            plot_dir = os.path.join(dest_dir, 'plots')
            shutil.rmtree(plot_dir, ignore_errors=True)
            os.makedirs(plot_dir, exist_ok=True)

    df, is_partial = get_dataframe(dataset_file, indexes, questions_ids, random_n)

    tokens_dependencies_extractor = config.tokens_dependencies_extractor
    tokens_dep_to_qdmr_extractor = config.tokens_dependencies_to_qdmr_extractor

    tok_deps_map = {}
    redecomp_map = {}
    errors_map = {}
    for index, row in df.iterrows():
        question_id, question_text, gold_decomposition, operators, *_ = row
        operators = [x.strip() for x in literal_eval(operators)]

        data_structures = {} if (plot_dir or func) else None
        error_message = None
        try:
            tok_dep = tokens_dependencies_extractor.extract(question_id=question_id,
                                                            question=question_text,
                                                            decomposition=gold_decomposition,
                                                            operators=operators,
                                                            debug=data_structures)
            if data_structures: data_structures['tokens_dependencies'] = tok_dep
            tok_deps_map[question_id] = tok_dep

            decomposition = tokens_dep_to_qdmr_extractor.extract(tok_dep, debug=data_structures)
            if data_structures: data_structures['decomposition'] = decomposition

            decomposition_str = decomposition.to_break_standard_string()
            assert decomposition_str, 'built an empty decomposition'
            if is_partial:
                print(f'{question_id}: {decomposition_str}')
            redecomp_map[question_id] = decomposition_str
        except Exception as ex:
            if is_partial:
                print(f'error in {question_id}: {str(ex)}')
            traceback.print_exc()
            error_message = str(ex)
            errors_map[question_id] = error_message
        if plot_dir:
            with open(os.path.join(plot_dir, f'{question_id}.html'), 'wt') as f:
                data_structures['question_id'] = question_id
                f.write(render_html(**data_structures, message=error_message))
        if func:
            params = {"question_id": question_id, "question_text": question_text, "gold_decomposition": gold_decomposition,
                      "operators":operators}
            params.update(**data_structures)
            func(**params)

    tok_deps = [tok_deps_map.get(x, 'ERROR') for x in df['question_id']]
    redecomp = [redecomp_map.get(x, 'ERROR') for x in df['question_id']]
    errors = [errors_map.get(x, '') for x in df['question_id']]

    preds_file = dest_dir and os.path.join(dest_dir,'dependencies_graph_redecomp.csv')

    if not is_partial and preds_file and os.path.exists(preds_file):
        # regression test
        old = pd.read_csv(preds_file)
        if len(old) == len(redecomp):
            old['decomposition_new'] = redecomp
            old['error_new'] = errors
            diff = old[old['decomposition'] != old['decomposition_new']]
            if len(diff) == 0:
                print('no changes')
            else:
                preds_file = os.path.splitext(preds_file)[0] + '-new.csv'
                print(f'{len(diff)} changes were detected. save decompositions to {preds_file}')
                print(f"{len(diff[diff['decomposition'] == 'ERROR'])} errors where fixed")
                print(f"{len(diff[diff['decomposition_new'] == 'ERROR'])} new errors")
                diff.to_csv(os.path.splitext(preds_file)[0] + '__diff.csv', index=True)

    redecomp_df = pd.DataFrame()
    redecomp_df['question_id'] = df['question_id']
    redecomp_df['question_text'] = df['question_text']
    redecomp_df['gold_decomposition'] = df['decomposition']
    redecomp_df['decomposition'] = redecomp
    redecomp_df['error'] = errors
    if dest_dir:
        redecomp_df.to_csv(preds_file, index=False)

    # evaluate
    # if is_eval:
    #     output_base = dest_dir and preds_file.replace('.csv', '_eval')
    #     eval_out = output_base and open(output_base+'.txt', 'w')
    #     if indexes:
    #         sys.stdout = eval_out or open(os.devnull, 'w')
    #         res = evaluate(
    #             ids=df['question_id'].to_list(),
    #             questions=df['question_text'].to_list(),
    #             golds=[format_qdmr(x) for x in df['decomposition']],
    #             decompositions=[format_qdmr(x) for x in redecomp],
    #             metadata=df,
    #             output_path_base=output_base,
    #             num_processes=5
    #         )
    #     else:
    #         if eval_out:
    #             sys.stdout = eval_out
    #         res = evaluate_predictions(dataset_file=dataset_file, preds_file=preds_file, output_file_base=output_base)
    #     sys.stdout = sys.__stdout__
    #     print('Eval:', res)

    if is_eval:
        eval_out = dest_dir and open(os.path.join(dest_dir, 'eval.txt'), 'w')
        sys.stdout = eval_out
        res = evaluate_dep_graph(dataset=df, dest_dir=dest_dir, predicted_graphs=tok_deps)
        sys.stdout = sys.__stdout__
        print('Eval:', res)

    return redecomp_df


def get_dataframe(dataset_file, indexes, questions_ids, random_n):
    """
    :param dataset_file:
    :param indexes:
    :param questions_ids:
    :param random_n:
    :return: dataframe, is partial mode
    """
    df = pd.read_csv(dataset_file)
    if questions_ids:
        indexes = df.index[df['question_id'].isin(questions_ids)].tolist()
        assert indexes, f"could not find {questions_ids}"
    elif indexes:
        indexes = [i - 2 for i in indexes]
    elif random_n:
        indexes = random.sample(range(len(df.index)), k=random_n)
    else:
        indexes = None
    if indexes:
        df = df.loc[indexes]
    return df, indexes is not None

def render_html(question_id: str,
                # QDMR dependencies graph
                steps_spans: StepsSpans = None,
                steps_dependencies: StepsDependencies = None,
                collapsed_spans_dependencies: SpansDependencies = None,
                tokens_dependencies: TokensDependencies = None,

                # dependencies graph to QDMR
                pre_unwind_spans_dependencies: SpansDependencies = None,
                spans_dependencies: SpansDependencies = None,
                decomposition: Decomposition = None,
                message: str = None,
                **kwargs):

    body = f"""
{message or ""}
<div>
<pre style="float:left; width:50%">{steps_spans and steps_spans.get_alignments_str()}</pre>
<pre style="float:left; width:50%">{decomposition and decomposition.to_break_standard_string(multiline=True)}</pre>
</div>
<div>
<figure>steps_dependencies{steps_dependencies and steps_dependencies.render_svg()}</figure>
<div>collapsed_spans_dependencies{collapsed_spans_dependencies and collapsed_spans_dependencies.render_html(level_reorder=False, include_properties=False)}</div>
</div>
<figure>{tokens_dependencies and tokens_dependencies.render_svg()}</figure>
<div>
<div>pre_unwind_spans_dependencies{pre_unwind_spans_dependencies and pre_unwind_spans_dependencies.render_html(include_properties=False)}</div>
<div>spans_dependencies{spans_dependencies and spans_dependencies.render_html()}</div>
</div>
    """
    template = utils_html.HTMLTemplate()
    return template.get_body(question_id=question_id, body=body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert gold decompositions to tokens dependencies and back to QDMR')
    parser.add_argument('-i', '--input-file', default='datasets/Break/QDMR/dev.csv',
                        help='dataset file to use (default: datasets/Break/QDMR/dev.csv)')
    parser.add_argument('-o', '--output-dir', type=str,
                        help='directory path for dependencies, plots and eval. if None, doesnt plot.')
    parser.add_argument('-p', '--plot', action="store_true",
                        help='whether to plot spans & dependencies')
    parser.add_argument('-q', '--question-id', nargs='+',
                        help='question_id(s) to use')
    parser.add_argument('-n', '--random-n', type=int,
                        help='sample n random questions')
    parser.add_argument('--no-eval', action='store_true', help='skip evaluation')
    args = parser.parse_args()

    import time
    start_time = time.time()
    to_dependencies_and_back_from_dataset(
        dataset_file=args.input_file,
        dest_dir=args.output_dir,
        plot_dependencies=args.plot,
        questions_ids=args.question_id,
        random_n=args.random_n,
        is_eval=not args.no_eval,
    )
    print(f'execution duration: {(time.time()-start_time)/60} minutes')
