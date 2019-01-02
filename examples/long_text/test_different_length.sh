rm 0.*
rm -rf log_dir/*
CUDA_VISIBLE_DEVICES="0" nohup python long_text_gen.py --present_rate 0.5 --max_seq_length 32  > 0.5.32.nohup.out &
CUDA_VISIBLE_DEVICES="1" nohup python long_text_gen.py --present_rate 0.5 --max_seq_length 64  > 0.5.64.nohup.out &
CUDA_VISIBLE_DEVICES="2" nohup python baseline.py --present_rate 0.5 --max_seq_length 32  > 0.5.32.seq2seq.nohup.out &
CUDA_VISIBLE_DEVICES="3" nohup python baseline.py --present_rate 0.5 --max_seq_length 64  > 0.5.64.seq2seq.nohup.out &
