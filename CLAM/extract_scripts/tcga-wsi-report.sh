#!/bin/bash
DIR_TO_COORDS="/data/path/to/patches/"
DATA_DIRECTORY="/data/path/to/WSIs"
CSV_FILE_NAME="/data/path/to/wsi_report_data.csv"
FEATURES_DIRECTORY=$DIR_TO_COORDS
ext=".tif"
save_storage="No"
root_dir="/data/path/to/logs/"

# models="resnet50"
#models="ctranspath"
# models="plip"
models="dinov2_vitl"

declare -A gpus
#gpus["resnet50"]=0
#gpus["resnet101"]=0
#gpus["ctranspath"]=0
gpus["dinov2_vitl"]=0
#gpus['plil']=0

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 16 \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage > $root_dir$model".log" 2>&1 & 

done