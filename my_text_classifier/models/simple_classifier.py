from typing import Dict
import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.F1 = FBetaMeasure()

    def forward( # forward的参数必须和instance的key值要一样，这样才可以对应去处理相应的内容，也可以自己在后续进行改造之类的
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            self.F1(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output
    
    # 这个应该是model里面原来就有的了, 这边相当于是重写, 而且这个是每个batch都会输出这个，最好是弄成每个epoch输出会比较好
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self.F1.get_metric(reset)
        output = {}
        output['accuracy'] = self.accuracy.get_metric(reset=reset)
        # 这里的counter是有很多个类别的意思, 这里我的想法，要不就是只输出总的precison, recall, fscore就好了
        counter, total_precison, total_recall, total_fscore = 0, 0, 0, 0
        for precision, recall, fscore in zip(f1_dict['precision'], f1_dict['recall'], f1_dict['fscore']):
            total_precison += precision
            total_recall += recall
            total_fscore = fscore
            counter += 1
        output['marco_precision'] = total_precison / counter
        output['marco_recall'] = total_recall / counter
        output['marco_fscore'] = total_fscore / counter
        return output
        # for precision, recall, fscore in zip(f1_dict['precision'], f1_dict['recall'], f1_dict['fscore']):
        #     output[str(counter) + '_precision'] = precision
        #     output[str(counter) + '_recall'] = recall
        #     output[str(counter) + '_fscore'] = fscore
        #     counter += 1
        # return output

        # return {"accuracy": self.accuracy.get_metric(reset), "F1":self.F1.get_metric(reset)}
