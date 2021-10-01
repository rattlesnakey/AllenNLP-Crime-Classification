from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer() # 前面满足的话，就不会用后面的了
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
    # text_to_intance这里label默认是None, 如果是predict的阶段的时候，就直接传text不会传label，所以label就是None, 所以和_read函数没有关系的
    # predict的时候是直接调用text_to_instance，没有调用到read, _read只有train和dev有用到
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                text, sentiment = line.strip().split("\t")
                yield self.text_to_instance(text, sentiment)
