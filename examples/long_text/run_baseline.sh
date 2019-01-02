rm 0.*.baseline.*
rm -rf log_dir/*present0.*.baseline
CUDA_VISIBLE_DEVICES="0" nohup python baseline.py --present_rate 0.2 > 0.2.baseline.nohup.out &
CUDA_VISIBLE_DEVICES="1" nohup python baseline.py --present_rate 0.5 > 0.5.baseline.nohup.out &
CUDA_VISIBLE_DEVICES="2" nohup python baseline.py --present_rate 0.8 > 0.8.baseline.nohup.out &

