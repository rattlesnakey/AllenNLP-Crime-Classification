local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/crime_dataset/train.tsv",
    "validation_data_path": "data/crime_dataset/valid.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder": {
            "type": "cnn",
            "embedding_dim": 768,
            "num_filters": 64,
            "ngram_filter_sizes": [2,3],
            "output_dim":64
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 1.0e-5
        },
        "cuda_device":-1,
        "num_epochs": 5
    }
}
