export PYTHONPATH=..:$PYTHONPATH

####################################
export CUDA_VISIBLE_DEVICES=4
DDP_WORLD_SIZE=1
DDP_PORT=12346

DATAROOT="/dev/shm/why_data/ProGAN"
CLASSES="airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse"
NAME="FreDect"
BS=12
####################################

cd ..
python -W ignore script/train.py \
--ddp_world_size $DDP_WORLD_SIZE \
--ddp_port $DDP_PORT \
--dataroot $DATAROOT \
--classes $CLASSES \
--name $NAME \
--detection_model $NAME \
--num_threads 8 \
--optim "adamw" \
--batch_size $BS