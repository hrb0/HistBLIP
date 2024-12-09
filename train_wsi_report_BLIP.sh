model='histgen'
max_length=100
epochs=70
region_size=96
prototype_num=768

CUDA_VISIBLE_DEVICES=5 python main_train_AllinOne.py \
    --image_dir /data2/yumi/kmedicon/patch_0920/pt_files/ \
    --ann_path /data2/yumi/kmedicon/annotation_ku677.json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --num_layers 5 \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --lr_ve 1e-4 \
    --lr_ed 1e-5 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 768 \
    --eos_idx 35233 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /data3/Thanaporn/kmedicon/Report/logs_BLIP_base_KU_patch_ep70_b10 \
    --step_size 1 \
    --gamma 0.8 \
    --seed 456789 \
    --log_period 50 \
    --beam_size 10 \
    --lr_scheduler LinearLR \
    --optim AdamW \
    --weight_decay 1e-4 \
    --model BLIP \
    --model_vis CONCH \
    


