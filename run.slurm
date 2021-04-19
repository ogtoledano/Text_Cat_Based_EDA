#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --partition=public
#SBATCH --mem=10G
#SBATCH --mem-per-cpu=1G
#SBATCH --nodelist=nodo[019]
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eda_topic_rec
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ogtoledano@uci.cu
#SBATCH --time=14-00

module load Python
source .bashrc

export PYTHONPATH=$PYTHONPATH:"/home/CLUSTER/uclv_ogtoledano/doctorado/Topic_Rec_Based_EDA"

srun -n 1 python /home/CLUSTER/uclv_ogtoledano/doctorado/Topic_Rec_Based_EDA/scripts/main_eda_cumda.py