#!/bin/bash -l


#SBATCH -D ./
#SBATCH --array=1-30

#SBATCH -o ./jobs_log/job_%A_%a.out        
#SBATCH -e ./jobs_log/job_%A_%a.err       


#SBATCH -J rotated_mnist
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

#SBATCH --mail-type=none
#SBATCH --mail-user=ali.gholamzadeh@tuebingen.mpg.de
#SBATCH --time=10:00:00

# Default mode
mode="train"

# Parse command-line arguments for mode
while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done


module purge 
module load anaconda/3/2023.03

source venv/bin/activate


srun python main.py --config=./configs/mnist/mnist.py --exp=rotated_mnist --mode=$mode