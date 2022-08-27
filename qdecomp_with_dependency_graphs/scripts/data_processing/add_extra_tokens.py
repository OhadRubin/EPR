from pathlib import Path
import pandas as pd
from os import path

from qdecomp_with_dependency_graphs.dependencies_graph.create_dependencies_graphs import get_extra_tokens


def main(root_dir: str):
    files = Path(root_dir).rglob('*_seq2seq.csv')
    extra_tokens, _ = get_extra_tokens()
    extra_tokens = ' '.join(extra_tokens)
    for fp in files:
        fp = str(fp)
        try:
            print(f'process {fp}...')
            df = pd.read_csv(fp)
            df['question_text'] = df['question_text'].apply(lambda x: f'{x} {extra_tokens}')
            dst_fp = path.splitext(fp)[0]+'__extra_tok.csv'
            df.to_csv(dst_fp, index=False)
        except Exception as ex:
            print(f'ERROR: {ex}')

if __name__ == '__main__':
    main(root_dir= 'datasets/Break/QDMR/')