# SeqGAN for Text Generation

This example is an implementation of [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf), with a language model as generator and a RNN-based classifier as discriminator.

Model structure and parameter settings are in line with SeqGAN in [Texygen](https://github.com/geek-ai/Texygen), except that we did not implement rollout strategy in discriminator for the consideration of simplicity.

Experiments are performed on COCO Captions, with 2k vocabularies and an average sentence length of 25. Both training and testing datasets contain 10k sentences.

## Usage

### Dataset
```shell
python data_utils.py --config config --data_path ./ --dataset coco
```

Here:
* `--config` specifies config parameters to use. Default is `config`.
* `--data_path` is the directory to store the downloaded dataset. Default is './'.
* `--dataset` indicates the training dataset. Currently `ptb` and `coco`(default) are supported.

### Train the model

Training can be performed with the following command:

```shell
python seqgan_train.py --config config --data_path ./ --dataset coco
```

Here:

`--config`, `--data_path` and `dataset` shall be the same with the flags settings used to download the dataset.

## Log



## Results

BLEU on image COCO caption train dataset:

|    |Texar - SeqGAN   | TexyGen - SeqGAN |
|---------------|-------------|----------------|
|BLEU1 | 0.744633 | 0.719211 |
|BLEU2 | 0.532205 | 0.446522 |
|BLEU3 | 0.297904 | 0.220235 |
|BLEU4 | 0.132374 | 0.082812 |

BLEU on image COCO caption test dataset:

|       | Texar - SeqGAN | TexyGen - SeqGAN |
| ----- | -------------- | ---------------- |
| BLEU1 | 0.566257       | 0.570906         |
| BLEU2 | 0.288737       | 0.265700         |
| BLEU3 | 0.120900       | 0.098115         |
| BLEU4 | 0.042441       | 0.028699         |





