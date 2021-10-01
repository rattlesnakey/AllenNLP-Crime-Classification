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
            "type": "lstm",
            "input_size":768,
            "hidden_size":128
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
        "num_epochs": 5,
        "cuda_device":-1
    }
}
