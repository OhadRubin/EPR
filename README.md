# Learning To Retrieve Prompts for In-Context Learning

Author implementation of this [NAACL 2022 paper](https://arxiv.org/abs/2112.08633).

## Training

To generate the training data using the LM and train the retriever use:
```
python run.py dataset={break|mtop|smcalflow} dpr_epochs=120 gpus=4 partition=killable no_slurm=True
```
To score using the OpenAI API use:
```
python api_scorer.py example_file=$EXAMPLE_FILE setup_type=qa output_file=$OUTPUT_FILE batch_size=1 +task_name=break engine=davinci-codex +n_shards=100 +shard_id=0
```
To run predictions with the OpenAI API use:
```
python api_client.py prompt_file=$PROMPT_FILE task_name=TASK_NAME output_file=$OUTPUT_FILE  engine=davinci-codex
```


## MISC
If more information is needed, please open an issue on this repo and let me know.