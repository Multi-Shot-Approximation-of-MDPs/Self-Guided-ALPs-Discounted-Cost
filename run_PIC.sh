#!/usr/bin/env bash

mdp_name='PIC'
algo_name='FALP'            #FALP, PG-FALP, SG-FALP
basis_func_type='fourier'   #relu, fourier, lns, stump
max_num_constr=200000
max_basis_num=20
batch_size=5
num_cpu_core=8
state_relevance_inner_itr=5

for instance_number in 1; do
    for seed in 111; do
    python main_PIC.py $mdp_name $algo_name $basis_func_type $instance_number $max_num_constr $max_basis_num \
         $batch_size $num_cpu_core $seed $state_relevance_inner_itr
    done
done