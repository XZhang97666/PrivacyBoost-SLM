start_idx=10
end_idx=1366
mode="context"
stop_token="/n/n"
dataset_name=medqa
split=dev
batch_prompt_size=1
echo 'dataset_name: '${dataset_name}
echo 'split: '${split}
echo 'start_idx: '${start_idx}
echo 'end_idx: '${end_idx}
echo 'mode: '${mode}
echo 'stop_token: '${stop_token}
python api_call.py \
--dataset_name ${dataset_name} \
--split ${split} \
--stop_token ${stop_token} \
--start_idx ${start_idx} \
--temperature 0 \
--prefix "cot_gpt3_5" \
--mode ${mode} \
--end_idx ${end_idx} \
--batch_prompt_size ${batch_prompt_size} \

