# Adapted from: https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_pt.sh

lr=2e-4
lora_rank=16
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/model/Llama-2-7b-hf
amharic_tokenizer_path=/model/llama-2-amharic-3784m

dataset_dir=./dataset/
data_cache=./cache/
per_device_train_batch_size=16
per_device_eval_batch_size=1
gradient_accumulation_steps=1
#warmup_steps=200


output_dir=./output/

python pretrain.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${amharic_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 7528 \
    --evaluation_strategy steps \
    --eval_steps 3000 \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --bf16 \
    --overwrite_output_dir \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --gradient_checkpointing \