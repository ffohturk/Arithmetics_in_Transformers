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

srun --unbuffered python Multiplication_generative_LG.py \
	--split 0.3 \
	--weight_decay 0.3 \
	--num_layers 6 \
	--d_model 512 \
	--d_ff 2048 \
	--ID 24 \
	--epochs 200 \
	--batch_size 32 \
	--batch_size_eval 512 \
	--learning_rate 1e-5 \
	--ndigits 3 \
	--nextra 3 \
	--pad 0 \
	# --output_dir multiplication_generative/
	# --pad 10 \
	
