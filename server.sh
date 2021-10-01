allennlp serve \
    --archive-path model/model.tar.gz \
    --predictor sentence_classifier \
    --field-name sentence \
    --include-package my_text_classifier \
    --host 202.112.194.62 \
    -p 8085
