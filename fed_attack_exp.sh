## block for smaller alexnet
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid --model_name covid --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=3 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin --model_name skin --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina --model_name retina --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --client_adversary 1 --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest --model_name chest --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney --model_name kidney --target_learning_rate 0.02 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 400 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --track_loss 1 --data_aug 2 --max_instance_per_batch 100 --attacker_epoch_gap 100  --user_number 1 --eval_data_size 1000 --target_data_size 20000 --num_step 200 --best_layer 4 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 150 --dataset cifar10 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --data_aug 0 --max_instance_per_batch 100 --attacker_epoch_gap 1  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 500 --dataset cifar100 --model_name densenet_cifar --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 250 375 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --track_loss 1 --data_aug 2 --max_instance_per_batch 100 --attacker_epoch_gap 100  --user_number 1 --eval_data_size 1000 --target_data_size 20000 --num_step 200 --best_layer 4 --client_adversary 1 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 150 --dataset cifar10 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

# resnet width experiment
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --track_loss 1  --data_aug 2 --max_instance_per_batch 100 --attacker_epoch_gap 500  --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 300 --dataset cifar10 --model_name resnet18 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 150 225 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500

## block for correct mobilenetv3
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py  --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid299 --model_name mobilenetv3 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name mobilenetv3 --target_learning_rate 0.5 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 100 --track_loss 0 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name mobilenetv3 --target_learning_rate 0.5 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest299 --model_name mobilenetv3 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ### chest good

#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --track_loss 1 --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney299 --model_name mobilenetv3 --target_learning_rate 0.5 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 400 ### 0.01 LR

#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --track_loss 1 --max_instance_per_batch 1 --attacker_epoch_gap 100  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 200 --dataset cifar100 --model_name mobilenetv3 --target_learning_rate 0.5 --target_l2_ratio 1e-5  --schedule 300 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

## block for correct alexnet
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 10 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid299 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 25 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name alexnet --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name alexnet --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest299 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=3 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney299 --model_name alexnet --target_learning_rate 0.02 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 400 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 1 --attacker_epoch_gap 1  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 300 --dataset cifar100 --model_name alexnet --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

## block for resnet50
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid299 --model_name resnet50 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name resnet50 --target_learning_rate 1e-3 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=3 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name resnet50 --target_learning_rate 1e-3 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest299 --model_name resnet50 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney299 --model_name resnet50 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 400 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 200  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset cifar100 --model_name resnet50 --target_learning_rate 1e-4 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

## block for inception
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid299 --model_name inception --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name inception --target_learning_rate 1e-3 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name inception --target_learning_rate 1e-3 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest299 --model_name inception --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ###  5e-3
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 200 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney299 --model_name inception --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 400 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=3 python3 fed_attack_exp.py --track_loss 0 --max_instance_per_batch 200 --attacker_epoch_gap 1  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset cifar100 --model_name inception --target_learning_rate 1e-4 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

## block for smaller densenet?
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 250 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 2000 --num_step 20 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100  --dataset covid299 --model_name densenet121 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000 ### 100 epochs
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name densenet121 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name densenet121 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 100 epochs

#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name densenet121 --target_learning_rate 0.1 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 100 --user_number 10 --eval_data_size 1000 --target_data_size 3000 --num_step 30 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset retina299 --model_name densenet121 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 1000 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 200 --target_data_size 200 --num_step 2 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset chest299 --model_name densenet121 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 200 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --server_name ramos --max_instance_per_batch 100 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 400 --target_data_size 400 --num_step 4 --best_layer 0 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 100 --dataset kidney299 --model_name densenet121 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 500 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 0.01 LR
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py --max_instance_per_batch 100 --attacker_epoch_gap 1  --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --best_layer 4 --client_adversary 0 --active_attacker_epoch 1 --active_attacker 0 --target_epochs 300 --dataset cifar100 --model_name densenet121 --target_learning_rate 0.01 --target_l2_ratio 1e-5  --schedule 150 225 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 1000

## block for rank test
#CUDA_VISIBLE_DEVICES=2 python3 fed_attack_exp.py --max_instance_per_batch 100 --track_loss 0 --random_seed 0 --test_rank 5 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 500 --target_data_size 500 --num_step 5 --active_attacker 0 --target_epochs 100 --dataset skin299 --model_name skin_special --target_learning_rate 0.5 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1 --validation_set_size 500 ### 0.01 LR, 0.05MMD

## block for binary purchase / texas
#CUDA_VISIBLE_DEVICES=0 python3 fed_attack_exp.py  --max_instance_per_batch 100 --signsgd 0 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --active_attacker 0 --target_epochs 100 --dataset texas --model_name texas --target_learning_rate 0.001 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1
#CUDA_VISIBLE_DEVICES=1 python3 fed_attack_exp.py --max_instance_per_batch 100 --signsgd 0 --track_loss 0 --attacker_epoch_gap 1 --user_number 10 --eval_data_size 1000 --target_data_size 4000 --num_step 40 --active_attacker 0 --target_epochs 100 --dataset purchase --model_name purchase --target_learning_rate 0.001 --target_l2_ratio 1e-5  --schedule 150 --local_epochs 1 --target_batch_size 100 --gpu 1

## defense param --dpsgd 0 --grad_norm 0 --noise_scale 0 --mixup 0 --mmd 0 --mmd_loss_lambda 0 --signsgd 0