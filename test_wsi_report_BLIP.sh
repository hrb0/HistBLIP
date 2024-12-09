model='histgen'
max_length=100
epochs=40
region_size=96
prototype_num=512

CUDA_VISIBLE_DEVICES=0 python main_test_AllinOne.py \
    --image_dir /data/path/to/pt_files/ \
    --ann_path /data/path/to/annotation.json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --num_layers 3 \
    --threshold 10 \
    --batch_size 1 \
    --epochs $epochs \
    --step_size 1 \
    --topk 512 \
    --cmm_size 2048 \
    --cmm_dim 768 \
    --eos_idx 35233 \
    --region_size $region_size \
    --prototype_num $prototype_num \
    --save_dir /data/path/to/Report/ \
    --step_size 1 \
    --gamma 0.8 \
    --seed 42 \
    --log_period 1000 \
    --load /data/path/to/logs/model_best.pth \
    --beam_size 3 \
    --d_model 768 \
    --d_ff 768 \
    --model_vis CONCH \
    --model BLIP


    # --image_dir /data2/yumi/kmedicon/patch_pu/pt_files/ \
    # --ann_path /data3/Thanaporn/kmedicon/annotation_public_dummy.json \