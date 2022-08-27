import argparse
import os
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt

def plot_metrics(src_dir:str, metrics:[str], output_dir: str = None):
    # get all metrics
    pathlist = Path(src_dir).glob("**/*metrics_epoch*.json")
    experiments_to_metrics = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for path in pathlist:
        parent = str(path.parent)
        if output_dir:
            parent = os.path.join(src_dir, output_dir, parent.replace(src_dir, ''))
        if parent not in experiments_to_metrics:
            experiments_to_metrics[parent] = []
        experiments_to_metrics[parent].append(str(path))

    # plot by experiment
    for expr, paths in experiments_to_metrics.items():
        jsons_data=[]
        for p in paths:
            with open(p, "rt") as f:
                data = json.load(f)
                jsons_data.append(data)
        df = pd.DataFrame.from_records(jsons_data).sort_values(["epoch"])

        base_dir = os.path.join(expr, "plots")
        os.makedirs(base_dir, exist_ok=True)

        # plot by metric
        for metric in metrics:
            try:
                ax = df.plot(x="epoch", y=metric, kind="line")
                ax.figure.savefig(os.path.join(base_dir, f"metrics_plot__{metric}"))
            except Exception as ex:
                print(f"Error in metric {metric} of {expr}: {str(ex)}")

        # todo: deal with too match opened figures
        #plt.close('all')  # very slow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="plot chosen metrics on all subdirectories"
                                                 "example: python scripts/plot_training_metrics.py --dir tmp -m training_loss -m validation_loss")
    parser.add_argument("--dir", type=str, help="root directory")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="destination directory. if not given, the plots will be saved in the experiment directory")
    parser.add_argument("-m", type=str, action="append", help="metrics to plot")
    args = parser.parse_args()
    assert os.path.exists(args.dir)

    plot_metrics(args.dir, args.m, args.output)