source activate newtf
rm 0.*
rm -rf log_dir/*present0.*
CUDA_VISIBLE_DEVICES="0" nohup python long_text_gen.py --present_rate 0.2 > 0.2.nohup.out &
CUDA_VISIBLE_DEVICES="1" nohup python long_text_gen.py --present_rate 0.5 > 0.5.nohup.out &
CUDA_VISIBLE_DEVICES="2" nohup python long_text_gen.py --present_rate 0.8 > 0.8.nohup.out &
