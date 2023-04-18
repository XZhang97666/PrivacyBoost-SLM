export CUDA_VISIBLE_DEVICES=0
model_name_or_path="t5-base"
dataset_name=obqa
shot=-1
seed=1
split='train_singletask_all_fid_qac_eA'
batch_size=8
# training and testing openbookqa

num_train_epochs=100
save_epoch_interval=1
num_warmup_steps=200
per_device_train_batch_size=$batch_size
gradient_accumulation_steps=1
 
python -W ignore run_t5_training_eval.py \
--model_name_or_path $model_name_or_path \
--dataset_name $dataset_name \
--seed $seed \
--save_epoch_interval $save_epoch_interval \
--num_train_epochs $num_train_epochs \
--num_warmup_steps $num_warmup_steps \
--per_device_train_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--shot $shot \
--train_split $split \
--eval_split qta_question_validation_qac_eA_$dataset_name \
--top_k 40 \
--eval_max_target_length 10 \
--temperature 1.0 \
--eval_mode "standard" \
--per_device_eval_batch_size 10 \
--test\
--output_dir ./debug_results/$input_dataset_name/shot_${shot}_split_${split}_model_${model_name_or_path}_seed_${seed}