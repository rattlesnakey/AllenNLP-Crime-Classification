set -v
set -e 
CONFIGURATION_PATH=../configuration/my_text_classifier_cnn.jsonnet
STORAGE_PATH=./checkpoints/bert-cnn-model
SRC_DIR=my_text_classifier

allennlp train ${CONFIGURATION_PATH} \
    -s ${STORAGE_PATH} \
    --include-package $SRC_DIR