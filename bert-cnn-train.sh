rm -rf bert-cnn-model
allennlp train my_text_classifier_cnn.jsonnet -s bert-cnn-model --include-package my_text_classifier