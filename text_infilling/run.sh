source activate newtf
rm 0.?.nohup.out
rm 0.5.32.nohup.out
rm -rf log_dir/*present0.*
# CUDA_VISIBLE_DEVICES="0" nohup python long_text_gen.py --present_rate 0.2 --batch_size 128 > 0.2.nohup.out &
CUDA_VISIBLE_DEVICES="1" nohup python long_text_gen.py --present_rate 0.5 --batch_size 200 > 0.5.nohup.out &
CUDA_VISIBLE_DEVICES="2" nohup python long_text_gen.py --present_rate 0.8 --batch_size 512 > 0.8.nohup.out &
CUDA_VISIBLE_DEVICES="3" nohup python long_text_gen.py --present_rate 0.5 --batch_size 100 --max_seq_length 32 > 0.5.32.nohup.out &
