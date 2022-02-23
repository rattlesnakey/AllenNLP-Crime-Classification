set -v
set -e
MODEL_PATH=X
TEST_DATA_PATH=X
SRC_DIR=my_text_classifier
allennlp evaluate ${MODEL_PATH} \
    ${TEST_DATA_PATH} \
    --include-package ${SRC_DIR} \
    --cuda-device 0