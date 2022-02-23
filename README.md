# AllenNLP-Crime-Classification
基于AllenNLP的罪名预测

# File Structure
```markdown
── configuration
│   ├── my_text_classifier_bert.jsonnet
│   ├── my_text_classifier_cnn.jsonnet
│   └── my_text_classifier_lstm.jsonnet
├── my_text_classifier
│   ├── dataset_readers
│   ├── __init__.py
│   ├── models
│   ├── predictors
│   └── __pycache__
├── README.md
├── requirements.txt
├── server.sh
└── train.sh
├── evaluate.sh
* configuration directory includes all the configure files
* my_text_classificatier directory includes the class which is registered
```

# Setup
`pip install -r requirements.txt`

# Train
`bash train.sh`

# Evaluate
`bash evaluate.sh`

# Start model as Server
`bash server.sh`

