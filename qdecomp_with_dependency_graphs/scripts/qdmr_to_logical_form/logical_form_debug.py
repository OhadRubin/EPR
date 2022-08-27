import os
import traceback
import pandas as pd
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.annotation_statistics import *
from qdecomp_with_dependency_graphs.scripts.data_processing.data_statistics import refs_count_per_operator

def qdmr_step_to_logical(step_text:str):
    step_identifier = StepIdentifier()
    try:
        step_text = parse_decomposition(str(step_text))[0]
        step = step_identifier.identify(step_text)
        return step.operator, str(step)
    except Exception as ex:
        traceback.print_exc()
        return "None", step_text

def to_logical_form(path: str):
    os.makedirs('_debug/logical_form', exist_ok=True)
    dest = os.path.join('_debug/logical_form', f'logical_form_{os.path.basename(path).split(".")[0]}.csv')
    ann = AnnotationEvaluator(path)
    ann.evaluate(dest)

def to_logical_form_per_step(path: str):
    os.makedirs('_debug/logical_form', exist_ok=True)
    dest = os.path.join('_debug/logical_form', f'logical_form_{os.path.basename(path).split(".")[0]}-per-step.csv')
    df = pd.read_csv(path)
    df = refs_count_per_operator(df)
    df['logical form'] = df['step'].apply(lambda x: qdmr_step_to_logical(x))
    df[['operator', 'logical form']] = pd.DataFrame(df['logical form'].tolist(), index=df.index)
    df.to_csv(dest, index=False)

def compare(set: str):
    df = pd.read_csv(f'_debug/logical_form/logical_form_{set}.csv')
    df_n = pd.read_csv(f'_debug/logical_form/logical_form_{set}-new.csv')
    df['program_new'] = df_n['program']
    diff = df['program']!=df['program_new']
    df_diff = df[diff]
    print(len(df_diff))
    # df_diff.to_csv(f'_debug/logical_form/logical_form_{set}-diff.csv', index=False)

def compare_steps(path: str):
    df = pd.read_csv(path)
    df['logical form new'] = df['step'].apply(lambda x: qdmr_step_to_logical(x))
    df[['operator new', 'logical form new']] = pd.DataFrame(df['logical form new'].tolist(), index=df.index)
    df = df[
        (df['logical form'] != df['logical form new']) |
        (df['operator'] != df['operator new'])
        ]
    diffs = len(df)
    if diffs:
        print(f'{diffs} differences')
        print((f"{len(df[df['logical form new'] == 'ERROR'])} new errors"))
        df.to_csv(os.path.splitext(path)[0] + '--diff.csv')
    else:
        print("no differences")


if __name__ == '__main__':
    devset = r'datasets/Break/QDMR/dev.csv'
    trainset = r'datasets/Break/QDMR/train.csv'

    # print(qdmr_step_to_logical("return which is bigger of #3 ,  #4"))
    to_logical_form(devset)
    # to_logical_form(trainset)

    # compare('dev')
    # compare('train')


    # refs_count_per_operator(pd.read_csv(devset)).to_csv('_debug/logical_form/dev-per-step.csv', index=False)
    # refs_count_per_operator(pd.read_csv(trainset)).to_csv('_debug/logical_form/train-per-step.csv', index=False)
    # compare_steps(f'_debug/logical_form/dev-per-step.csv')
    # compare_steps(f'_debug/logical_form/train-per-step.csv')

    to_logical_form_per_step(devset)
    # to_logical_form_per_step(trainset)
    # compare_steps(f'_debug/logical_form/logical_form_dev-per-step.csv')
    # compare_steps(f'_debug/logical_form/logical_form_train-per-step.csv')


