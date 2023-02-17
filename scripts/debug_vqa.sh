
VQA_EXP="/home/mila/s/sarvjeet-singh.ghotra/scratch/models/uniter/debug"
TXT="/home/mila/s/sarvjeet-singh.ghotra/scratch/data/img/train/vqa/uniter/txt_db"
IMG="/home/mila/s/sarvjeet-singh.ghotra/scratch/data/img/train/vqa/uniter/img_db/"

cd ..


CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.run --master_port 29501 \
    -m debugpy --connect 127.0.0.1:5678 \
    inf_vqa.py \
    --txt_db $TXT/vqa_test.db \
    --img_db $IMG/coco_test2015 \
    --output_dir $VQA_EXP \
    --checkpoint 6000 \
    --pin_mem \
    --fp16


# horovodrun -np 4 python train_vqa.py --config config/train-vqa-base-4gpu.json \
#     --output_dir $VQA_EXP
