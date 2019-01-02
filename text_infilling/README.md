# Instructions for Varying Mask Rates and #Blanks #

This set of experiments study the impact of the mask rate (percentage of masked tokens) and the number of blanks on model performance. The mask positions and lengths are selected randomly according to the desired mask rate and #blanks. 



The Yelp review corpus is used for training and testing:

- Training/testing dataset: 104K/1K sentences
- Vocabulary size: 9K



## Usage

### Dataset

Download the dataset with the following command:

```bash
python data_utils.py
```

Yelp data will be download and stored in `yelp_data/`.


### Train and Test the model

Training can be performed with the following command:

```bash
python [MODEL].py --mask_rate [MASK_RATE] --blank_num [BLANK_NUM] --filename_prefix 'pos.' --data_dir './yelp_data/pos/'
```

Here:

- `MODEL` is the model to train. May be `self_attn`, `seq2seq` or `gan`.
- `MASK_RATE` specifies the portion of words masked out in the template.` 
- `BLANK_NUM` specifies the number of blanks in the template.



## Results

### Evaluation Results

The following table displays the quantitative and human evaluations results when removing 30%, 40% and 50% of the tokens in the template. With the same mask rate, we test the generation process with templates containing one or two blanks.

| Mask Rate     | #Blanks     | Metric     | Template     | Seq2Seq     | GAN     | Self-attn     |
| ------------- | ----------- | ---------- | ------------ | ----------- | ------- | ------------- |
| 30%           | 1           | BLEU       | 63.916       | 69.097      | 68.470  | **71.104**    |
| 30%           | 1           | Perplexity | -            | 107.480     | 144.127 | **38.304**    |
| 30%           | 1           | Human Eval | -            | 1.950       | 1.775   | **2.275**     |
| 30%           | 2           | BLEU       | 42.233       | 67.174      | 64.337  | **65.914**    |
| 30%           | 2           | Perplexity | -            | 43.044      | 36.704  | **21.028**    |
| 30%           | 2           | Human Eval | -            | 1.838       | 1.975   | **2.188**     |
| **Mask Rate** | **#Blanks** | **Metric** | **Template** | **Seq2Seq** | **GAN** | **Self-attn** |
| 40%           | 1           | BLEU       | 56.838       | 61.309      | 61.778  | **63.543**    |
| 40%           | 1           | Perplexity | -            | 202.714     | 230.569 | **44.864**    |
| 40%           | 1           | Human Eval | -            | **2.075**   | 1.865   | 2.055         |
| 40%           | 2           | BLEU       | 38.279       | 55.460      | 55.326  | **59.192**    |
| 40%           | 2           | Perplexity | -            | 59.877      | 70.195  | **25.914**    |
| 40%           | 2           | Human Eval | -            | 2.005       | 1.900   | **2.045**     |
| **Mask Rate** | **#Blanks** | **Metric** | **Template** | **Seq2Seq** | **GAN** | **Self-attn** |
| 50%           | 1           | BLEU       | 44.369       | 48.865      | 48.861  | **51.55**     |
| 50%           | 1           | Perplexity | -            | 244.862     | 287.415 | **43.688**    |
| 50%           | 1           | Human Eval | -            | 1.838       | 1.975   | **2.412**     |
| 50%           | 2           | BLEU       | 32.498       | 42.613      | 42.535  | **44.418**    |
| 50%           | 2           | Perplexity | -            | 99.421      | 107.558 | **32.397**    |
| 50%           | 2           | Human Eval | -            | 1.875       | 1.913   | **2.238**     |



### Infilling Results

An example model outputs on a Yelp test case, where the template contains two blanks and 40% of the tokens are masked out:

| Template     | I live \_\_m\_\_ and I was _\_m\_\_ chinese food .           |
| ------------ | ------------------------------------------------------------ |
| Ground Truth | i live <u>right down the street</u> and i <u>was craving some good</u> chinese food . |
| Seq2seq      | i live <u>at a ten times</u> and i was <u>at appreciated by</u> chinese food . |
| GAN          | i live <u>right of the app</u> and i was <u>looking for chinese food .</u> |
| Self-attn    | i live <u>in the neighborhood area</u> and i was <u>impressed with the chinese food .</u> |



