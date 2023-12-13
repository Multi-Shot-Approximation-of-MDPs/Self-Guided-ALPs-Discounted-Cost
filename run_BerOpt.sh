#!/usr/bin/env bash

VFA_num_train_sample_path=1000
VFA_num_test_sample_path=5000
inner_sample_size=100
state_relevance_type='lognormal' 
abs_val_upp_bound='1000'
compute_IR_bound='True'
max_basis_num=20
batch_size=10
preprocess_batch=50
basis_func_type='fourier'   #relu, fourier, lns, stump
CFA_num_train_sample_path=10000
CFA_num_test_sample_path=10000
IR_inner_sample_size=100
num_cpu_core=16

for instance_number in 1; do   
    for seed in 111; do
        
        # Prepare random basis functions
        python main_BerOpt_ALP.py 'preprocess' $instance_number $seed $VFA_num_train_sample_path $VFA_num_test_sample_path $inner_sample_size\
                                     $state_relevance_type $abs_val_upp_bound\
                                         $compute_IR_bound $basis_func_type $max_basis_num $batch_size $num_cpu_core $preprocess_batch
        
        #Run LSMN algorithm 
        python main_BerOpt_LSM.py $instance_number  $CFA_num_train_sample_path\
                         $CFA_num_test_sample_path $seed $num_cpu_core $IR_inner_sample_size
                                        
        #Run ALP + DFM algorithm 
        #python main_BerOpt_ALP.py 'FALP' $instance_number $seed $VFA_num_train_sample_path $VFA_num_test_sample_path $inner_sample_size\
        #                    $state_relevance_type $abs_val_upp_bound\
        #                     $compute_IR_bound 'DFM' $max_basis_num $batch_size $num_cpu_core $preprocess_batch
                                        
        # Run FALP
        #python main_BerOpt_ALP.py 'FALP' $instance_number $seed $VFA_num_train_sample_path $VFA_num_test_sample_path $inner_sample_size\
        #                            $state_relevance_type $abs_val_upp_bound\
        #                                $compute_IR_bound $basis_func_type $max_basis_num $max_basis_num $num_cpu_core $preprocess_batch
                                        
        # Run SG-FALP
        #python main_BerOpt_ALP.py 'SG_FALP' $instance_number $seed $VFA_num_train_sample_path $VFA_num_test_sample_path $inner_sample_size\
        #                    $state_relevance_type $abs_val_upp_bound\
        #                        $compute_IR_bound $basis_func_type $max_basis_num $batch_size $num_cpu_core $preprocess_batch
                    
    done
done


