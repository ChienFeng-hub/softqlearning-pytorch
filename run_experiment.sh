export CUDA_VISIBLE_DEVICES=0,1,2
declare -a env_names=("HalfCheetah-v4" "Ant-v4" "Humanoid-v4" "Hopper-v4"  "Walker2d-v4")
config=$algo
wandb=0
steps=5000000
N_SEEDS=5
description='standard'

cnt=0
for ((i=0;i<N_SEEDS;i+=1))
do
    for env in "${env_names[@]}"
    do

        device="cuda:$cnt"
        (( cnt=(cnt+1)%3 ))

        python train.py env=$env config=$env seed=$i steps=$steps description=$description wandb=$wandb device=$device &

    done
done
wait