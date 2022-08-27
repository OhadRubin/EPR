# Question Decomposition with Dependency Graphs

Code for the dependency-based [QDMR](https://allenai.github.io/Break/) parsers described in our paper 
["Question Decomposition with Dependency Graphs"](https://arxiv.org/abs/2104.08647) (by Matan Hasson and Jonathan Berant). 
It is based on [AllenNLP](https://allennlp.org/) and pytorch.

## Structure
* [dependencies_graph](dependencies_graph) - conversion procedures from QDMRs to Dependency Graphs and vice versa; 
  and Logical-Form based evaluation metric.
* [evaluation](evaluation) - Break official evaluation metrics, including Normalized-Exact-Match.
* [experiments](experiments) - AllenNLP experiments configurations, including the [best](experiments/_best) ones. 
* [qdecomp_nlp](qdecomp_nlp) - modules implementations, following the AllenNLP structure convention.
* [scripts](scripts) - scripts for data creation, training, evaluation, etc.

## Environment Setup
To setup a virtual environment with the relevant packages, please run:
```angular2html
conda create -n qdecomp python=3.8.5
conda activate qdecomp

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"
```
Note: we use the conda environment name in some of our training/evaluation scripts. 

## Data Creation
### Download and Preprocess BREAK
Our seq2seq based models use a formatted version of the BREAK dataset files.
1. [Download](https://github.com/allenai/Break/raw/master/break_dataset/Break-dataset.zip) the dataset files.
1. Place them in ```datasets/```, so you have ```datasets/Break/QDMR/{dev, train, test}.csv```
2. Run the preprocess script:
   ```angular2html
    python scripts/data_processing/preprocess_examples.py datasets/Break/QDMR --parse
   ```
### Dependency Graphs Creation
Our data files for the dependency graphs are published [here](datasets.zip). \
To recreate them, please follow these steps:
1. Place BREAK dataset files in ```datasets/```, so you have ```datasets/Break/QDMR/{dev, train, test}.csv```
2. Extract the steps spans (token alignment):
   ```angular2html
   DEP_CONF=special_tokens_ILP python dependencies_graph/create_steps_spans.py extract --data datasets/Break/QDMR/dev.csv
   DEP_CONF=special_tokens_ILP python dependencies_graph/create_steps_spans.py extract --data datasets/Break/QDMR/train.csv
   ```
3. Extract dependency graphs:
    ```angular2html
    python dependencies_graph/create_dependencies_graphs.py 
    ```
   For the ```dev``` and ```test``` sets we create files with the questions only, so one can predict the entire set 
   (including samples with no dependency graph),
   and evaluate using a metric which is not relay on the gold graphs (s.a. the Logical-Form Exact Match).

For domain/sample generalization, please use the [partial data script](scripts/data_processing/partial_data.py).

## Train Models
Our configurations for the discussed models in the paper are placed in [experiments/](experiments), 
where the final configurations are given [experiments/_best](experiments/_best).

To run an experiment (train + predict + eval), please run:
```angular2html
CUDA_VISIBLE_DEVICES=0 python scripts/train/run_experiments.py train --experiment experiments/<path-to-experiment>.jsonnet -s <output-root-dir>
```
Specifically, train & evaluate the final models by:
```angular2html
# CopyNet+BERT
CUDA_VISIBLE_DEVICES=0 python scripts/train/run_experiments.py train -s tmp --experiment experiments/_best/Break/QDMR/seq2seq/copynet--transformer-encoder.json

# BiaffineDG
CUDA_VISIBLE_DEVICES=0 python scripts/train/run_experiments.py train -s tmp --experiment experiments/_best/Break/QDMR/dependencies_graph/biaffine-graph-parser--transformer-encoder.json

# Latent-RAT
CUDA_VISIBLE_DEVICES=0 python scripts/train/run_experiments.py train -s tmp --experiment experiments/_best/Break/QDMR/hybrid/multitask--copynet--latent-rat-encoder_separated.json
```

A dedicated execution directory will be ceated for each experiment in ```<output-root-dir>```, 
and the predictions and the evaluation files will be placed in ```<experiment-out-dir>/eval```. \
Note: the graph models use the `dev_dependencies_graph.json` data file, which does not cover the whole dev set. 
For full prediction/evaluation, you can either change the configuration to use `dev_dependencies_graph__questions_only.json`
or use it in the flag `--dataset_file` of `scripts/train/run_experiments.py eval`. 


## Evaluation

Our script for running the experiments also evaluates the model's predications. \
However, in case you want to run the evaluation independently:
```
python scripts/eval/evaluate_predictions.py \
--dataset_file datasets/Break/QDMR/{dev,test}.csv  \
--preds_file <path-to-predictions> \
--output_file_base <output-files-prefix> \
```

The predictions file, `preds_file`, is assumed to be a CSV file with a `decomposition` column 
containing a model's *predictions*, ordered according to `dataset_file`.
The `output_file_base` indicates the file to which the evaluation output be saved. 
You can control the evaluation metrics to apply using the `metrics` flag. \
To use the `*_preds.json` files that are generated by our `run_experiments` script 
(or `allennlp predict` command) as a `preds_file`, please add the `--allennlp` flag. 

### LF-EM Evaluation
To get more detailed evaluation, including the compared logical-forms, please use
[this script](dependencies_graph/evaluation/evaluate_dep_graph.py).

## Supplementary Tools

* Render model's predicted DGs (html files) - [examine_predictions](dependencies_graph/examine_predictions.py), `render` command.
* Convert DGs to QDMRs - [examine_predictions](dependencies_graph/examine_predictions.py), `qdmr` command.
* [ILP decoder](dependencies_graph/run_ILP_based_decoder.py) for BiaffineGP.
* Normalized Exact Match metric [implementation](evaluation/normal_form).
* [Creation](scripts/data_processing/create_decomp_logical_form.py) of LFs dataset.
* [Summarization](scripts/eval/eval_summarize_experiments_scores.py) of all the eval files of the experiments in a single csv file.
* Optuna based tuning [scripts](scripts/tune).
