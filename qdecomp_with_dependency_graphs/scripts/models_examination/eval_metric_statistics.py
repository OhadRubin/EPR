from typing import List, Dict, Set

import os
import re
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from nltk import ngrams


def plot_overlap(names: List[str], eval_dfs: List[str], metric_col: List[str], title=None):
    assert len(names) == len(eval_dfs) == len(metric_col)
    df = pd.DataFrame()
    for n, path, metric in zip(names, eval_dfs, metric_col):
        df[n] = pd.read_csv(path)[metric]

    def merge_label(row):
        correct = [x for x in names if row[x]]
        if not correct: return "None"
        if len(correct) == len(names): return "All"
        return ','.join(sorted(correct))

    df['label'] = df.apply(lambda row: merge_label(row), axis=1)
    fig = px.pie(df, names='label', title=title)
    # fig.show()
    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=0,
    #     xanchor="center",
    #     x=0,
    # ))

    # fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_layout(
        autosize=False,
        width=600,
        height=300,
        legend=dict(
                # orientation="h",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.75,
        )
    )

    fig.write_image(os.path.join(DEST, f'models_overlap-{",".join(sorted(names))}.png'))


import spacy
parser = spacy.load('en_core_web_sm', disable=['ner'])
def plot_accuracy(gold:str, names: List[str], eval_dfs: List[str], metric_col: List[str]):
    assert len(names) == len(eval_dfs) == len(metric_col)
    df = pd.read_csv(gold)
    for n, path, metric in zip(names, eval_dfs, metric_col):
        df[n] = pd.read_csv(path)[metric]

    df['sub-dataset'] = df['question_id'].apply(lambda x: x.split('_')[0])
    df['steps'] = df['decomposition'].apply(lambda x: len(x.split(';')))
    df['question_length'] = df['question_text'].apply(lambda x: len(parser.tokenizer(x)))

    for g in ['sub-dataset', 'steps', 'question_length']:
        df_ = df.groupby(g).mean()
        fig = go.Figure(data=[
            go.Bar(name=n, x=df_.index, y=df_[n])
            for n in names
        ])
        fig.update_layout(barmode='group', yaxis=dict(range=[0, 1]))
        # fig.show()
        fig.update_layout(
            legend=dict(
                # orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        fig.write_image(os.path.join(DEST, f'accuracy-{g}--{",".join(sorted(names))}.png'))


def calculate_ngrams(gold_lf:List[str], names: List[str], eval_dfs: List[str], metric_col: List[str], N=None):
    # LF chains
    chains: List[List[List[str]]] = []
    errors = 0
    for _, lf in enumerate(gold_lf):
        has_error = False
        try:
            steps = lf.split('\n')
            steps_chains: Dict[str, List[List[str]]] = {}
            for i, x in enumerate(steps):
                s_chains: List[List[str]] = []
                try:
                    x = re.sub(r'^\s*\d+\.\s*', '', x)
                    operator, prop, args = re.match(r'(\w+)(?:\[(.*)\])?\((.*)\)\s*', x).groups()
                    # re.match(r'(\w+)(?:\[(\w*)\])?\((?:(\w+=[^;]*)[;]?)+\)', 'select(sub=abc)').groups()
                    args = [a.split('=') for a in args.split(';')]
                    for name, val in args:
                        refs =[int(r) for r in re.findall(r'#(\d+)', val)]
                        s_chains.extend(c+[f'{operator}-{name}[{prop}]']
                                        for r in refs
                                        for c in (steps_chains.get(r) or [[]]))
                    if not s_chains:
                        s_chains.append([f'{operator}-[{prop}]'])
                except Exception as ex:
                    print(f"Error in lf {lf}: step {i}: {x}. "+str(ex))
                    has_error=True
                steps_chains[i+1] = s_chains
        except Exception as ex:
            print(f"Error in lf {lf}: "+str(ex))
            has_error = True
        errors += int(has_error)
        chains.append(steps_chains[len(steps)])
    print(f"errors: {errors/len(gold_lf)*100}% ({errors}/{len(gold_lf)})")

    # extend ngrams
    if N:
        for c_list in chains:
            for chain in list(c_list):
                grams = [ list(x)
                          for n in range(1, N)
                          for x in ngrams(chain, n)
                          ]
                c_list.extend(grams)

    # chains to sets of str
    chains_set: List[Set[str]] = [set(','.join(c) for c in c_list) for c_list in chains ]
    chains_to_metric = defaultdict(lambda: defaultdict(lambda: 0))
    for n, path, metric in zip(names, eval_dfs, metric_col):
        vals = pd.read_csv(path)[metric]
        for v, c_set in zip(vals, chains_set):
            inc = 1 if v else 0
            for c in c_set:
                chains_to_metric[c][n]+=inc

    df = pd.DataFrame.from_records(data=[{'chain':k, **v} for k,v in chains_to_metric.items()])
    df.to_csv(os.path.join(DEST, f'{N}-ngrams-{",".join(sorted(names))}.csv'))

    # filter + order
    df_norm = df.set_index("chain")
    lengths = df["chain"].apply(lambda x: len(x.split(','))).to_list()
    max_val = df_norm.max(axis=1)
    df_norm = df_norm.div(df_norm.max(axis=1), axis=0).fillna(0)
    min_val = df_norm.min(axis=1)
    df_norm['lengths'] = lengths
    df_norm['original_max'] = max_val
    df_norm['min'] = min_val
    df_norm.reset_index(inplace=True)

    df_filtered = df_norm[df_norm['original_max']>0].sort_values(by=['min', 'original_max'], ascending=[True, False])
    df_filtered.to_csv(os.path.join(DEST, f'{N}-ngrams-{",".join(sorted(names))}__filtered-and-ordered.csv'))


def check_ensemble():
    copy_df = pd.read_csv('tmp/_best/Break/QDMR/seq2seq/copynet--transformer-encoder/copynet--transformer-encoder/eval/datasets_Break_QDMR_dev_seq2seq__eval_full.csv')
    lat_df = pd.read_csv('tmp/_best/Break/QDMR/hybrid/multitask--copynet--latent-rat-encoder_separated/multitask--copynet--latent-rat-encoder_separated/eval/datasets_Break_QDMR_dev_seq2seq__eval_full.csv')
    copy_df.columns = [f'{x}_copy' for x in copy_df.columns]
    lat_df.columns = [f'{x}_lat' for x in lat_df.columns]
    df = pd.concat([copy_df, lat_df], axis=1)
    def final_lf(row):
        steps = row['gold_copy'].split('@@SEP@@')
        if len(steps) >= 5:
            return row['logical_form_em_lat']
        return row['logical_form_em_copy']
    df['logical_form_em'] = df.apply(final_lf, axis=1)
    df_ = df[['logical_form_em', 'logical_form_em_copy','logical_form_em_lat']].mean().round(3)
    print(df_)

if __name__ == '__main__':
    DEST = 'tmp/analysis'
    os.makedirs(DEST, exist_ok=True)

    evals = [
        ('CopyNet+BERT', 'logical_form_em', 'tmp/_best/Break/QDMR/seq2seq/copynet--transformer-encoder/copynet--transformer-encoder/eval/datasets_Break_QDMR_dev_seq2seq__eval_full.csv'),
        ('BiaffineGP', 'logical_form_tokens_exact_match', 'tmp/_best/Break/QDMR/dependencies_graph/biaffine-graph-parser--transformer-encoder/biaffine-graph-parser--transformer-encoder/eval/dev_preds_eval.csv'),
        ('Latent-RAT', 'logical_form_em', 'tmp/best/Break/QDMR/hybrid/multitask--copynet--latent-rat-encoder_separated/multitask--copynet--latent-rat-encoder_separated/eval/datasets_Break_QDMR_dev_seq2seq__eval_full.csv'),
    ]
    # plot_overlap(
    #     names=[x[0] for x in evals],
    #     eval_dfs=[x[2] for x in evals],
    #     metric_col=[x[1] for x in evals],
    # )
    #
    # plot_accuracy(
    #     gold='datasets/Break/QDMR/dev.csv',
    #     names=[x[0] for x in evals],
    #     eval_dfs=[x[2] for x in evals],
    #     metric_col=[x[1] for x in evals],
    # )

    # gold_lfs = list(pd.read_csv(evals[1][2])['gold_logical_form_tokens_original'])
    # calculate_ngrams(
    #     gold_lf=gold_lfs,
    #     names=[x[0] for x in evals],
    #     eval_dfs=[x[2] for x in evals],
    #     metric_col=[x[1] for x in evals],
    #     N=25
    # )

    check_ensemble()
