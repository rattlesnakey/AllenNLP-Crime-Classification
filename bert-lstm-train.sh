rm -rf bert-lstm-model
allennlp train my_text_classifier_lstm.jsonnet -s bert-lstm-model --include-package my_text_classifier