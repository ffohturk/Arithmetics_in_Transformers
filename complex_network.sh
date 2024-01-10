#!/bin/bash

#SBATCH --job-name=projectX
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=kruthoff@ias.edu
#SBATCH --export=all

###master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
###export MASTER_ADDR=$master_addr

module load anaconda3/
source activate ML

srun torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:25400 Addition_complex_network.py \
	--split 0.3 \
	--weight_decay 0.3 \
	--num_layers 2 \
	--ID 0 \
	--epochs 1000 \
	--batch_size 1024 \
	--learning_rate 1e-3 \
	--output_dir Complex_Network/
