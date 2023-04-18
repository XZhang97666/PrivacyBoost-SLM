#directly on gpt results
model_name_or_path="stanford-crfm/BioMedLM"  
dataset_name=medqa
input_dataset_name=medqa

########################MedQA########################
task='_all_qac_eA'
split='train_singletask'$task
lr=2e-6
shot=-1
accelerate launch --gpu_ids "2,3" \
    run_gpt_training_eval.py \
    --dataset_name $dataset_name \
    --seed $seed \
    --model_name_or_path $model_name_or_path \
    --shot $shot \
    --eval_split qta_question_validation_${task}_$dataset_name\
    --per_device_eval_batch_size 10 \
    --test\
    --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_seed_${seed}_lr_${lr}

########################HEADQA########################
dataset_name=headqa
input_dataset_name=headqa

task='_all_qac_eA'
split='train_singletask'$task
lr=2e-6
shot=-1
accelerate launch --gpu_ids "2,3" \
    run_gpt_training_eval.py \
    --dataset_name $dataset_name \
    --seed $seed \
    --model_name_or_path $model_name_or_path \
    --shot $shot \
    --eval_split qta_question_validation_${task}_$dataset_name\
    --per_device_eval_batch_size 10 \
    --test\
    --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_seed_${seed}_lr_${lr}




########################MedMCQA########################

dataset_name=medmcqa
input_dataset_name=medmcqa

task='_all_qac_eA'
split='train_singletask'$task
lr=2e-6
shot=10000
accelerate launch --gpu_ids "2,3" \
    run_gpt_training_eval.py \
    --dataset_name $dataset_name \
    --seed $seed \
    --model_name_or_path $model_name_or_path \
    --shot $shot \
    --eval_split qta_question_validation_${task}_$dataset_name\
    --per_device_eval_batch_size 10 \
    --test\
    --output_dir ./results/$input_dataset_name/shot_${shot}_split_${split}_seed_${seed}_lr_${lr}