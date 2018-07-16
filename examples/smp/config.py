data_dir = "./toy/"
vocab_file = data_dir + 'vocab.txt'
train_data_hparams = {
    "num_epochs": 123,
    "batch_size": 23,
    "datasets": [
        {  # dataset 0: sentences
            "files": [data_dir + 'sentences.txt'],
            "vocab_file": vocab_file,
            "data_name": "sentence"
        },
        {  # dataset 1: labels
            "files": data_dir + 'labels.txt',
            "data_type": "int",
            "data_name": "label"
        },
    ]
}
