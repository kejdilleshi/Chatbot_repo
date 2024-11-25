#!/bin/bash --login
#SBATCH --job-name LLM
#SBATCH --error LLM-%j.error
#SBATCH --output LLM-%j.out
#SBATCH --time 0-24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --account rfabbret_llm

module load gcc/11.4.0 cuda/11.8.0 cudnn/8.7.0.84-11.8

conda activate NLP



# papermill My_chatbot.ipynb My_chatbot_out.ipynb
python GPT2.py --results_dir "ensemble_results/SamBot_1-1-3" 
