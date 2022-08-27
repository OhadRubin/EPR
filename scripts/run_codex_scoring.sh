
for i in $(seq 1 99); do
	python api_scorer.py example_file=$PWD/data/bm25_break-null_a_train.json setup_type=qa \
	output_file=$PWD/data/bm25_break-null+scr_model-codex_a_train_scoredqa_pre_$i.json \
	batch_size=1 +task_name=break engine=davinci-codex +n_shards=100 +shard_id=$i ; done
