#!/bin/bash
#SBATCH --job-name=Tiamat_Optuna_HPC
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=15
#SBATCH --mem=16G
#SBATCH --nodelist=work1
#SBATCH --time=00-01:00:00
#SBATCH --output=logs/saida_%j.out
#SBATCH --error=logs/erro_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

 
# Execução correta chamando o python:

python3 /home/leonardo2610065/aprendizado_de_maquina/tiamat/tiamat_optuna_hpc.py \
    --data data/biodeg.csv \
    --n-trials 150 \
    --output resultados_hpc
