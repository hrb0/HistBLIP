export OPENCV_IO_MAX_IMAGE_PIXELS=10995116277760

source_dir="/data3/Thanaporn/kmedicon/WSIs"
wsi_format="tif"

patch_size=512
save_dir="/data3/Thanaporn/kmedicon/dinov2_vitl"
python create_patches_fp.py \
    --source $source_dir \
    --save_dir $save_dir\
    --preset tcga.csv \
    --patch_level 0 \
    --patch_size $patch_size \
    --step_size $patch_size \
    --wsi_format $wsi_format \
    --seg \
    --patch \
    --stitch 
