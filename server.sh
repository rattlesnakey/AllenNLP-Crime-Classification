set -v
set -e
PORT=8085
HOST=x
MODEL_PATH=

allennlp serve \
    --archive-path ${MODEL_PATH} \
    --predictor sentence_classifier \
    --field-name sentence \
    --include-package my_text_classifier \
    --host ${HOST} \
    -p ${PORT}
