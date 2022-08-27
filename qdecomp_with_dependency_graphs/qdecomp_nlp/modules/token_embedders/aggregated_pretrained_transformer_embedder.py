from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder


@TokenEmbedder.register("aggregated_pretrained_transformer")
class AggregatedPretrainedTransformerEmbedder(PretrainedTransformerEmbedder):
    """
    Gets tokens from basic tokenizer (i.e no wordpieces), re-tokenize each token and aggregate
    its sub-tokens embedding.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vocab_namespace: str,
                 model_name: str,
                 **kwargs):
        assert 'max_length' not in kwargs, 'max_length is not allowed'
        super().__init__(model_name=model_name, **kwargs)
        self.vocab = vocab
        self.vocab_namespace = vocab_namespace
        self._tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)
        self._indexer = PretrainedTransformerIndexer(model_name)

    def map_special_tokens(self, source_token: str):
        # todo: depends on transformer model? tokens?
        if source_token == self.vocab._padding_token:
            return self._tokenizer.tokenizer._pad_token  # '[PAD]'
        if source_token == START_SYMBOL:
            return self._tokenizer.tokenizer._cls_token  # '[CLS]'
        if source_token == END_SYMBOL:
            return self._tokenizer.tokenizer._sep_token  # '[SEP]'
        return source_token

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.size(0)
        embeddings_list = []

        for i in range(batch_size):
            current_tokens = tokens[i]

            # index (of basic tokenizer) to tokens
            index_to_token = self.vocab.get_index_to_token_vocabulary(namespace=self.vocab_namespace)
            tokens_text = [self.map_special_tokens(index_to_token[x]) for x in current_tokens.tolist()]

            # re-tokenize tokens
            new_tokens = []
            start_indices = []
            for token_text in tokens_text:
                start_indices.append(len(new_tokens))
                sub_tokens = self._tokenizer.tokenize(token_text)
                new_tokens.extend(sub_tokens)

            # embed
            indices = self._indexer.tokens_to_indices(tokens=new_tokens, vocabulary=self.vocab)
            fixed_mask = [1 if t.text != self._tokenizer.tokenizer._pad_token else 0 for t in new_tokens]
            embeddings = super().forward(token_ids=torch.tensor(indices["token_ids"]).unsqueeze(0).to(device=tokens.device),  # Batch size 1,
                                         # mask=torch.tensor(indices["mask"]).unsqueeze(0).to(device=tokens.device),  # Batch size 1,
                                         mask=torch.tensor(fixed_mask).unsqueeze(0).to(device=tokens.device),  # Batch size 1,
                                         type_ids=torch.tensor(indices["type_ids"]).unsqueeze(0).to(device=tokens.device),  # Batch size 1,
                                         segment_concat_mask=None)

            # aggregate
            start_embeddings = embeddings.squeeze()[start_indices]
            embeddings_list.append(start_embeddings)

        return torch.stack(embeddings_list)