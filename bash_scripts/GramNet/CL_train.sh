export PYTHONPATH=..:$PYTHONPATH

####################################
export CUDA_VISIBLE_DEVICES=2

DATAROOT="CL_datasets/train"
DATASET_NAME="progan,deepfake,biggan,stylegan2,ddpm,adm,dalle,glide,sdv4,midjourney,vqdm,sdv2.1,sdxl1.0,sdv3.0"
NAME="GramNet"
WEIGHT_PATH="weights/classifier/GramNet/Gram.pth"
CL_METHODS=("Seq" "ER" "EWC" "OSLA" "AGEM" "SI" "iCaRL" "Linear")

BS=256
####################################

cd ..

for CL_METHOD in "${CL_METHODS[@]}"; do
    echo "Running with CL method: $CL_METHOD"
    python -W ignore script/CL_train.py \
    --dataroot $DATAROOT \
    --dataset_name $DATASET_NAME \
    --multiclass 1 0 0 1 0 0 0 0 0 0 0 0 0 0 \
    --detection_model $NAME \
    --initial_weight_path  $WEIGHT_PATH \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --cl_method $CL_METHOD \
    --skip_initial_training True \
    --output_dir ./outputs/continual_learning/$NAME/$CL_METHOD
done
