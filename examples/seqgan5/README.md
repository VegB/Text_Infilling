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


## Log

### Training loss
Training loss will be recoreded in log_dir/log.txt.
```text
G pretrain epoch   0, step 1: train_ppl: 81.639235
G pretrain epoch   1, step 1: train_ppl: 9.845531
G pretrain epoch   2, step 1: train_ppl: 7.581516
...
G pretrain epoch  78, step 1: train_ppl: 3.753437
G pretrain epoch  79, step 1: train_ppl: 3.711618
D pretrain epoch   0, step 0: dis_total_loss: 16.657263, r_loss: 8.789272, f_loss: 7.867990
D pretrain epoch   1, step 0: dis_total_loss: 3.317280, r_loss: 1.379951, f_loss: 1.937329
D pretrain epoch   2, step 0: dis_total_loss: 1.798969, r_loss: 0.681685, f_loss: 1.117284
...
D pretrain epoch  78, step 0: dis_total_loss: 0.000319, r_loss: 0.000009, f_loss: 0.000310
D pretrain epoch  79, step 0: dis_total_loss: 0.000097, r_loss: 0.000009, f_loss: 0.000088
G update   epoch  80, step 1: mean_reward: -56.315876, expect_reward_loss:-56.315876, update_loss: 9194.217773
D update   epoch  80, step 0: dis_total_loss: 0.000091, r_loss: 0.000008, f_loss: 0.000083
G update   epoch  81, step 1: mean_reward: -56.507019, expect_reward_loss:-56.507019, update_loss: 10523.346680
D update   epoch  81, step 0: dis_total_loss: 0.000230, r_loss: 0.000008, f_loss: 0.000222
...
G update   epoch 178, step 1: mean_reward: -58.171032, expect_reward_loss:-58.171032, update_loss: 15077.129883
D update   epoch 178, step 0: dis_total_loss: 0.000073, r_loss: 0.000003, f_loss: 0.000070
G update   epoch 179, step 1: mean_reward: -58.190083, expect_reward_loss:-58.190083, update_loss: 14430.581055
D update   epoch 179, step 0: dis_total_loss: 0.000019, r_loss: 0.000003, f_loss: 0.000016
```

### BLEU
BLEU1~BLEU4 scores will calculated every 10 epoches, the results are written to log_dir/bleu.txt.
```text
epoch 170 BLEU1~4 on train dataset:
0.726647
0.530675
0.299362
0.133602

 epoch 170 BLEU1~4 on test dataset:
0.548151
0.283765
0.118528
0.042177
```

