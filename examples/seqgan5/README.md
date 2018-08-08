# SeqGAN for Text Generation

This example builds a VAE for text generation, with LSTM as encoder and LSTM or Transformer as decoder. Training is performed on official PTB data and Yahoo data, respectively. Yahoo dataset is from [(Yang, et. al.) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139), which is created by sampling 100k documents from the original Yahoo Answer data. The average document length is 78 and the vocab size is 200k. 

This example is an implementation of [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf), with a language model as generator and a RNN-based classifier as discriminator.

## Usage

Training can be performed with the following command:

```shell
python vae_train.py --config config_trans_ptb --dataset ptb
```

Here:

* `--config` specifies the config file to use. If the dataset cannot be found in the specified path, dataset will be downloaded automatically, the downloading directory can be specified by `--data_path` (default is `./`)
* `--dataset` specifies the dataset to use, currently `ptb` and `yahoo` are supported

## Log



## Results

 BLEU on image COCO caption test dataset:

|    |Texar - SeqGAN   | TexyGen - SeqGAN |
|---------------|-------------|----------------|
|BLEU1 | 0.742512 | 0.77511 |
|BLEU2 | 0.557184 | 0.556697 |
|BLEU3 | 0.359806 | 0.335516 |
|BLEU4 | 0.195522 | 0.168896 |

