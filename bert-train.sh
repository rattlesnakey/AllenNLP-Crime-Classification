rm -rf bert-model
allennlp train my_text_classifier_bert.jsonnet -s bert-model --include-package my_text_classifier