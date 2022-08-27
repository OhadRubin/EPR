
import datasets
"""
domain in {'alarm',
 'calling',
 'event',
 'messaging',
 'music',
 'news',
 'people',
 'recipes',
 'reminder',
 'timer',
 'weather'}
"""
    


logger = datasets.logging.get_logger(__name__)
_URL = "https://fb.me/mtop_dataset"

_CITATION = """ """

_DESCRIPTION = """ """



class MtopConfig(datasets.BuilderConfig):
    """BuilderConfig for Mtop."""

    def __init__(self, **kwargs):
        """BuilderConfig for Mtop.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MtopConfig, self).__init__(**kwargs)


class Mtop(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        MtopConfig(
            name="mtop",
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
                    "intent": datasets.Value("string"),
                    "spans": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "lang": datasets.Value("string"),
                    "logical_form": datasets.Value("string"),
                    "tokenized_question": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath,"split":"train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": filepath,"split":"eval"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": filepath,"split":"test"}),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        
        with open(f"{filepath}/mtop/en/{split}.txt", encoding="utf-8") as f:
            for example in f:
                example = example.split("\t")
                dict_example = dict(idx=example[0],
                                    intent=example[1],
                                    spans=example[2],
                                    question=example[3],
                                    domain=example[4],
                                    lang=example[5],
                                    logical_form=example[6],
                                    tokenized_question=example[7])
                yield key, dict_example
                key += 1