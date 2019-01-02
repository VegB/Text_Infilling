# Instructions for Infilling Showcases #

Here we provide three showcases for Text Infilling:

Experiments are performed on two datasets:

- Grimm's Fairy Tale

- NBA Script



## Preposition Infilling

### Dataset

Download the dataset with the following command:

```python
python data_utils.py --dataset 'grimm_prep'
```

The dataset will be stored in `grimm_prep_data/`.




### Train and Test the model

Training can be performed with the following command:

```python
python [MODEL].py --filename_prefix 'grimm.prep.' --data_dir './grimm_prep_data/'
```

 `MODEL` flag specifies the model to train. It be `self_attn`, `seq2seq` or `gan`.



### Infilling Results

An example from the Grimm's Fairy Tales data where prepositions are masked out:

| Template     | \_\_m\_\_ old woman went _\_m\_\_ , but saw _\_m\_\_ one on the stairs |
| ------------ | ------------------------------------------------------------ |
| Ground Truth | <u>the</u> old woman went <u>out</u> , but saw <u>no</u> one on the stairs |
| Seq2seq      | <u>the</u> old woman went <u>with</u> , but saw <u>at</u> one on the stairs |
| GAN          | <u>the</u> old woman went <u>for</u> , but saw <u>no</u> one on the stairs |
| Self-attn    | <u>the</u> old woman went <u>in</u> , but saw <u>that</u> one on the stairs |



## Longer Content Infilling

### Download Dataset

Download the dataset with the following command:

```python
python data_utils.py --dataset [DATASET]
```

Here the `DATASET` flag may be `grimm_lm` or `nba_lm`.



### Train and Test the model

```python
python [MODEL].py --filename_prefix '[DATASET].lm.' --data_dir './[DATASET]_lm_data/'
```

Here:

- `Model` is the model to train. May be `self_attn`, `seq2seq` or `gan`.
- `Dataset` refers to the dataset for training. Set to  `grimm` for Grimm's Fairy Tale or `nba` for NBA scripts.



### Infilling Results

An example for language models with anchor words on Grimm Tales.



| Template     | \_\_m\_\_ sound _\_m\_\_ be _\_m\_\_                         |
| ------------ | ------------------------------------------------------------ |
| Ground Truth | <u>if you bear it without letting a</u> sound <u>escape you , i shall</u> be <u>free</u> |
| Seq2seq      | <u>and</u> sound <u>the</u> be <u>and the little , and the little , and the</u> |
| GAN          | <u>and</u> sound <u>the</u> be <u>and the , and and</u>      |
| Self-attn    | <u>the</u> sound <u>said , i will</u> be <u>the king</u>     |



An example of the NBA reports for language models with anchor words.

| Template     | _\_m\_\_ Toronto_Raptors _\_m\_\_ 114-110 _\_m\_\_           |
| ------------ | ------------------------------------------------------------ |
| Ground Truth | <u>The</u> Toronto_Raptors <u>defeated the Detroit_Pistons</u> 114 - 110 <u>on Sunday at the Air Canada</u> |
| Seq2seq      | <u>The</u> Toronto_Raptors <u>defeated the the</u> 114 - 110 <u>on Wednesday at the Center</u> |
| GAN          | <u>The</u> Toronto_Raptors <u>defeated the visiting</u> 114 - 110 <u>on Friday .</u> |
| Self-attn    | <u>The</u> Toronto_Raptors <u>defeated the Philadelphia_76ers</u> 114 - 110 <u>on Friday .</u> |

