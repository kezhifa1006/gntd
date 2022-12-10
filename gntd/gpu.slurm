#!/bin/bash
#SBATCH -J dqn            # 任务名字是 gpu-job
#SBATCH -o gpu.out         # target_soft;reward_decay=1;eps_decay;
#SBATCH --cpus-per-task=4     # 申请 8 个 cpu 核心，这里防止内存不够必须增加 CPU 核心数
#SBATCH -p gpu                # 任务提交到 gpu 分区
#SBATCH --gres=gpu:1          # 申请 1 块 GPU 卡，必须填写，SLURM 默认是不分配 GPU 的
#SBATCH -t 6-00:00:00         # 最长运行时间是 1 天 

module purge                  # 先清除用户自定义模块

module add mesa gcc/7.5.0          # 载入 gcc
module add anaconda/3         # 载入 anaconda 模块

source activate rl_env # 载入 conda 环境 

# 'CartPole-v1' 'Acrobot-v1' 'LunarLander-v2' 

envs=(
    'CartPole-v1'
    'Acrobot-v1'
    'LunarLander-v2' 
)


# policy='Expert' # 有使用normalize state，其他都没有
# policy='Random'
# policy='Medium-Replay'
# policy='Replay'
policy='Mix'

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		python main.py --train_offline --use_gntd --damping 0.5 --momentum --lr 1e-5 --env $env --seed $i --save_mat --policy $policy
        python main.py --train_offline --lr 1e-5 --env $env --seed $i --save_mat --policy $policy
        python main.py --train_offline --option td --use_gntd --damping 0.5 --momentum --lr 1e-5 --env $env --seed $i --save_mat --policy $policy
        python main.py --train_offline --option td --lr 1e-5 --env $env --seed $i --save_mat --policy $policy
	done
done
