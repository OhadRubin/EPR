from pathlib import Path
from shutil import copyfile, copytree, errno, ignore_patterns
import argparse
import os


def copy(exp_path:str, dest_path:str):
    patterns: [str] = ["**/evals", "**/plots", "**/config.json", "**/metrics.json"]
    exclude_patterns : [str] = ["*_preds.json", "*_summary.tsv"]
    experiments = [str(p.parent) for p in Path(exp_path).glob("**/evals/")]

    for exp in experiments:
        for pattern in patterns:
            exclude = [p for ex_patt in exclude_patterns for p in Path(exp).glob(ex_patt)]
            pathlist = [p for p in Path(exp).glob(pattern) if p not in exclude]
            for path in pathlist:
                 path_in_str = str(path)
                 d=os.path.join(dest_path, path_in_str)
                 os.makedirs(os.path.dirname(d), exist_ok=True)
                 print("{} -> {}".format(path_in_str, d))
                 try:
                     copytree(path_in_str, d, ignore=ignore_patterns(*exclude_patterns))
                 except OSError as exc:  # python >2.5
                     if exc.errno == errno.ENOTDIR:
                         copyfile(path_in_str, d)
                     else:
                         raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="copy aside evaluations files")
    parser.add_argument('--exp_dir', type=str, help='path to experiments directory')
    parser.add_argument('--dest_dir', type=str, help='path to destination directory')
    args = parser.parse_args()
    assert os.path.exists(args.exp_dir)
    assert args.exp_dir != args.dest_dir

    copy(args.exp_dir, args.dest_dir)