CUDA_VISIBLE_DEVICES=3 python3 client.py --user_id 0 --dataset cifar10 --model_name resnet18 --num_step 45 --target_learning_rate 0.1 --target_data_size 4500 --target_epochs 200 --target_batch_size 100 --target_l2_ratio 5e-4

#for i in `seq 0 1`; do
#  echo "Starting client $i"
#  python client.py --user_id ${i} --dataset cifar10 --model_name resnet18 --num_step 20 --target_learning_rate 0.1 --target_data_size 2000 --target_epochs 200 --target_batch_size 100 --target_l2_ratio 5e-4
#done

# This will allow you to use CTRL+C to stop all background processes
#trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
#wait