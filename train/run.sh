CUDA_VISIBLE_DEVICES=0 nohup python main.py --task BS --train --test --seed 2023 --run_id $1 > $1.log &
