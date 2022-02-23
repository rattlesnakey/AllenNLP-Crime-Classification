local bert_model = "bert-base-chinese";

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
    "train_data_path": "../data/crime_dataset/final_crime_data/filter_train_processed.tsv",
    "validation_data_path": "../data/crime_dataset/final_crime_data/filter_valid_processed.tsv",
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
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "type":"gradient_descent",
        "cuda_device":3,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps":1000
        },
        "num_epochs": 100,
        "patience":7,
        "num_gradient_accumulation_steps":2,
        "callbacks":[
            {
                "type":"wandb", 
                "project":"crime-classification",
                "entity":"hengyuan",
                "name":"bert-lstm-lr1e-5-warmup1000-allennlp",
                "watch_model":0,
                "should_log_learning_rate":1
            },
            {
                "type":"track_epoch_callback"
            }
        ]
    },
    "serialization_dir":"bert-lstm-model",
}
