python RL_assignment.py --model 2 --chunk_size 200 --learning_rate 3e-4 --batch_size 128 --buffer_size 1000000 --tau 0.0001 --gamma 0.99 --net_arch 128,128 --reward 4 --output_dir "./logs"
python RL_assignment.py --model 0 --chunk_size 200 --learning_rate 3e-4 --batch_size 64 --buffer_size 100000 --tau 0.0001 --gamma 0.90 --ent_coef -1.0 --net_arch 512,512 --reward 4 --output_dir "./logs"
python RL_assignment.py --model 1 --chunk_size 200 --learning_rate 1e-3 --batch_size 256 --gamma 0.90 --net_arch 256,256 --ent_coef 0.0 --reward 4 --output_dir "./logs"
