import argparse
import json
import os
import pandas as pd

from qdecomp_with_dependency_graphs.scripts.data_processing.preprocess_examples import process_target


def load_input_files(input_decompositions_file, input_spider_dir):
    decompositions = pd.read_csv(input_decompositions_file)
    decompositions[["dataset", "split", "example_idx"]] = decompositions.question_id.apply(
        lambda x: pd.Series(x.split("_", 2)))
    spider_decomp = decompositions[decompositions.dataset == "SPIDER"].to_dict(orient="records")
    for i in range(len(spider_decomp)):
        spider_decomp[i]["example_idx"] = int(spider_decomp[i]["example_idx"])

    spider_train_file = os.path.join(input_spider_dir, "train_spider.json")
    with open(spider_train_file, "r") as fd:
        spider_train = json.load(fd)

    spider_dev_file = os.path.join(input_spider_dir, "dev.json")
    with open(spider_dev_file, "r") as fd:
        spider_dev = json.load(fd)

    return spider_decomp, spider_train, spider_dev


def create_new_dataset(spider_decomp, spider_train, spider_dev):
    train_decomp, dev_decomp = [], []
    train_qs, dev_qs = [], []
    for decomp in spider_decomp:
        if decomp["split"] == "train":
            example = spider_train[decomp["example_idx"]]
        elif decomp["split"] == "dev":
            example = spider_dev[decomp["example_idx"]]
        else:
            continue

        assert example["question"] == decomp["question_text"]
        example_copy = json.loads(json.dumps(example))
        example["question"] = process_target(decomp["decomposition"])

        if decomp["split"] == "train":
            train_decomp.append(example)
            train_qs.append(example_copy)
        elif decomp["split"] == "dev":
            dev_decomp.append(example)
            dev_qs.append(example_copy)
        else:
            # not supposed to get here.
            continue

    return train_decomp, dev_decomp, train_qs, dev_qs


def write_outputs(train, dev, output_dir, decomp):
    file_name = "train_decomp_spider.json" if decomp else "train_qs_spider.json"
    train_file = os.path.join(output_dir, file_name)
    with open(train_file, "w") as fd:
        json.dump(train, fd)

    file_name = "dev_decomp.json" if decomp else "dev_qs.json"
    dev_file = os.path.join(output_dir, file_name)
    with open(dev_file, "w") as fd:
        json.dump(dev, fd)

    print(train_file)
    print(dev_file)


def main(args):
    spider_decomp, spider_train, spider_dev = load_input_files(
        args.input_decompositions_file, args.input_spider_dir)

    train_decomp, dev_decomp, train_qs, dev_qs = create_new_dataset(spider_decomp, spider_train, spider_dev)
    print("done!\n")
    print(f"converted {len(train_decomp)}/{len(spider_train)} train examples "
          f"and {len(dev_decomp)}/{len(spider_dev)} dev examples.")

    write_outputs(train_decomp, dev_decomp, args.output_dir, decomp=True)
    write_outputs(train_qs, dev_qs, args.output_dir, decomp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess examples")
    parser.add_argument('input_decompositions_file', type=str, help='path to input file')
    parser.add_argument('input_spider_dir', type=str, help='path to spider dataset directory')
    parser.add_argument('output_dir', type=str, help='path to output directory')
    args = parser.parse_args()
    assert os.path.exists(args.input_decompositions_file)
    assert os.path.exists(args.input_spider_dir)
    assert os.path.exists(args.output_dir)

    main(args)

