


import datasets

    
import json

logger = datasets.logging.get_logger(__name__)
_URL = "https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/raw/master/datasets/SMCalFlow%202.0/{}.dataflow_dialogues.jsonl.zip"
_URLS = {"train":_URL.format("train"),
        "validation":_URL.format("valid")}
_CITATION = """ """

_DESCRIPTION = """ """



class SmcalflowConfig(datasets.BuilderConfig):
    """BuilderConfig for Smcalflow."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smcalflow.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SmcalflowConfig, self).__init__(**kwargs)


class Smcalflow(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SmcalflowConfig(
            name="smcalflow",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("string"),
                    "user_utterance": datasets.Value("string"),
                    "lispress": datasets.Value("string"),
                    "fully_typed_lispress": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        filepath = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath["train"], "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": filepath["validation"], "split": "valid"}),
        
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        
        with open(f"{filepath}/{split}.dataflow_dialogues.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                idx = obj['dialogue_id']
                for example in obj['turns']:

                    dict_example = dict(user_utterance=example['user_utterance']['original_text'],
                                        lispress=example['lispress'],
                                        fully_typed_lispress=example['fully_typed_lispress'],
                                        idx=f"{idx}_{example['turn_index']}")
                    yield key, dict_example
                    key += 1