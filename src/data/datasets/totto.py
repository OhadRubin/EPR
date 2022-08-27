import json
import datasets

class ToTToDataset:
    def __init__(self):
        _URL = "https://storage.googleapis.com/totto-public/totto_data.zip"
        dl_manager = datasets.utils.download_manager.DownloadManager()
        self.cache_path = dl_manager.download_and_extract(_URL)
        self.splits = {}
        for split_name in ["train","dev"]:
            with open(f"{self.cache_path}/totto_data/totto_{split_name}_data.jsonl", 'r') as f:
                proccessed_dataset = []
                for example in f:
                    dict_example = json.loads(example)
                    proccessed_dataset.append(dict_example)
                self.splits[split_name] = proccessed_dataset