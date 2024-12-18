#CUDA_VISIBLE_DEVICES=2 python train.py \
#--name NPR-20classes \
#--dataroot ./datasets/ProGAN \
#--classes airplane,bird,bottle,car,cat,chair,horse,diningtable,person,sheep,train,bicycle,boat,bus,cow,dog,motorbike,pottedplant,sofa,tvmonitor \
#--batch_size 320 \
#--delr_freq 10 \
#--lr 0.0002 \
#--niter 50

CUDA_VISIBLE_DEVICES=2 python train.py \
--name NPR-4classes \
--dataroot ./datasets/ProGAN \
--classes car,cat,chair,horse \
--batch_size 320 \
--delr_freq 10 \
--lr 0.0002 \
--niter 50