CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30300 \
basicsr/train.py -opt $1 --launcher pytorch