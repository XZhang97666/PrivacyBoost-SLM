export CUDA_VISIBLE_DEVICES=2
model_name_or_path="michiyasunaga/BioLinkBERT-base"  # "razent/SciFive-base-Pubmed_PMC"

shot=-1
########################MedQA########################
dataset_name=medqa 
input_dataset_name=medqa

task='_all_qac_eA'
split='train_singletask'$task
batch_size=8
lr=5e-5
num_train_epochs=100
save_epoch_interval=1
num_warmup_steps=200
per_device_train_batch_size=$batch_size
gradient_accumulation_steps=1
seed=1
python -W ignore run_bert_training_eval.py \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --seed $seed \
        --num_train_epochs $num_train_epochs \
        --num_warmup_steps $num_warmup_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --shot $shot \
        --train_split $split \
        --eval_split qta_question_validation${task}_${dataset_name}\
        --per_device_eval_batch_size 10 \
        --train \
        --test\
        --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_model_${model_name_or_path}_seed_${seed}_lr_${lr}


########################HEADQA########################
dataset_name=headqa 
input_dataset_name=headqa

task='_all_qac_eA'
split='train_singletask'$task
batch_size=8
lr=5e-5
num_train_epochs=100
save_epoch_interval=1
num_warmup_steps=200
per_device_train_batch_size=$batch_size
gradient_accumulation_steps=1
seed=1
python -W ignore run_bert_training_eval.py \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --seed $seed \
        --num_train_epochs $num_train_epochs \
        --num_warmup_steps $num_warmup_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --shot $shot \
        --train_split $split \
        --eval_split qta_question_validation${task}_${dataset_name}\
        --per_device_eval_batch_size 10 \
        --train \
        --test\
        --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_model_${model_name_or_path}_seed_${seed}_lr_${lr}

########################MedMCQA########################

dataset_name=medmcqa
input_dataset_name=medmcqa

task='_all_qac_eA'
split='train_singletask'$task
batch_size=8
lr=2e-5
num_train_epochs=100
save_epoch_interval=1
num_warmup_steps=200
per_device_train_batch_size=$batch_size
gradient_accumulation_steps=1
seed=1
# shot=10000
python -W ignore run_bert_training_eval.py \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --seed $seed \
        --num_train_epochs $num_train_epochs \
        --num_warmup_steps $num_warmup_steps \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --shot $shot \
        --train_split $split \
        --eval_split qta_question_validation${task}_${dataset_name}\
        --per_device_eval_batch_size 10 \
        --train \
        --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_model_${model_name_or_path}_seed_${seed}_lr_${lr}