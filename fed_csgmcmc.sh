#CUDA_VISIBLE_DEVICES=3 python3 fed_csgmcmc.py --user_number 10 --target_data_size 1000 --num_step 10 --target_epochs 200 --dataset intel --model_name intel --target_learning_rate 0.01 --target_l2_ratio 5e-4  --schedule 300 --local_epochs 1 --target_batch_size 100 --gpu 1
CUDA_VISIBLE_DEVICES=2 python3 fed_csgmcmc.py --user_number 10 --target_data_size 4000 --num_step 40 --target_epochs 200 --dataset cifar10 --model_name resnet18 --target_learning_rate 0.1 --target_l2_ratio 5e-4  --schedule 300 --local_epochs 1 --target_batch_size 100 --gpu 1
