rm ptb.*
rm -rf ptb_log.*
python data_utils.py --config config_ptb_small
python data_utils.py --config config_ptb_medium
python data_utils.py --config config_ptb_large
CUDA_VISIBLE_DEVICES="0" nohup python seqgan_train.py --config config_ptb_small --dataset ptb > ptb.small.nohup.out &
CUDA_VISIBLE_DEVICES="1" nohup python seqgan_train.py --config config_ptb_medium --dataset ptb > ptb.medium.nohup.out &
CUDA_VISIBLE_DEVICES="2" nohup python seqgan_train.py --config config_ptb_large --dataset ptb > ptb.large.nohup.out &
