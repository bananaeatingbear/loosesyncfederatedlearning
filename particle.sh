#CUDA_VISIBLE_DEVICES=2 python3 particle.py --local_epochs 1 --num_step 5 --user_number 10 --target_data_size 2000 --target_epochs 300 --dataset cifar10 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --target_batch_size 100 --gpu 1
#CUDA_VISIBLE_DEVICES=1 python3 particle.py --local_epochs 3 --num_step 20 --user_number 10 --target_data_size 2000 --target_epochs 300 --dataset cifar100 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=1 python3 particle.py --num_step 20 --user_number 10 --target_data_size 2000 --target_epochs 150 --dataset fashion_mnist --model_name fashion_mnist --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=0 python3 particle.py --local_epochs 1 --user_number 10 --target_data_size 2000 --num_step 5 --target_epochs 100 --dataset retina --model_name retina --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 150 --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=2 python3 particle.py --local_epochs 1 --user_number 10 --target_data_size 2000 --num_step 1 --target_epochs 150 --dataset fashion_product_gender --model_name fashion_product --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 150 --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=3 python3 particle.py --local_epochs 1 --user_number 10 --target_data_size 2000 --num_step 20 --target_epochs 250 --dataset fashion_product_season --model_name fashion_product --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 250  --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=3 python3 particle.py --user_number 10 --target_data_size 2000 --num_step 1 --target_epochs 150 --dataset sat6 --model_name sat6 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1

CUDA_VISIBLE_DEVICES=2 python3 particle.py --user_number 10 --target_data_size 1000 --num_step 1 --target_epochs 100 --dataset intel --model_name intel --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1

#CUDA_VISIBLE_DEVICES=1 python3 particle.py --local_epochs 1 --user_number 10 --target_data_size 2000 --num_step 20 --target_epochs 50 --dataset mnist --model_name mnist --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 250  --target_batch_size 100 --gpu 1
