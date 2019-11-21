#! /bin/bash
cd "$(dirname "$0")"

cuda_device_id=0
device="cuda:"$cuda_device_id

basedir='/home/poolc2/Workspace/nip/kd/data/'

coef_act_fn=softmax
model_type=stochastic_linear_basis_cust
version_log=$model_type
cust_teacher_logit_path=Teachers/linear_basis_cust.coef_act_fn_softmax.num_bases_5.attribute_dim_64.key_query_size_64/Logits/logits_train.npy
non_cust_teacher_logit_path=Teachers/non_cust.det/Logits/logits_train.npy
reg_kd=0.005
num_bases=5
attribute_dim=64
key_query_size=64
subdir=$model_type".reg_kd_"$reg_kd
python3 -W ignore train.py \
--random_seed 9716 \
--cust_teacher_logit_path $basedir$cust_teacher_logit_path \
--non_cust_teacher_logit_path $basedir$non_cust_teacher_logit_path \
--reg_kd $reg_kd \
\
--n_experiments 1 \
\
--attribute_dim $attribute_dim \
--key_query_size $key_query_size \
--word_dim 300 \
--state_size 256 \
--num_bases $num_bases \
\
--model_type $model_type \
\
--subdir $subdir \
\
--version_log $version_log \
--device $device \
\
--batch_size 32 \
--eval_batch_size 32 \
\
--max_epochs 8 \
--eval_step 1000 \
--max_grad_norm 3.0 \
--uncertainty_method "" \
--std_update "false" \
--ent_update "false" \
\
--data_path_prefix $basedir \
--vocab_path $basedir"42939.vocab" \
--pretrained_word_em_dir $basedir"word_vectors.npy"
