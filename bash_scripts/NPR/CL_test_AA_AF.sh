export PYTHONPATH=..:$PYTHONPATH

####################################
export CUDA_VISIBLE_DEVICES=3

DATAROOT="CL_datasets/test"
CHECKPOINT_NAME="progan,deepfake,biggan,stylegan2,ddpm,adm,dalle,glide,sdv4,midjourney,vqdm,sdv2.1,sdxl1.0,sdv3.0"
DATASET_NAME="progan,cyclegan,stargan,deepfake,biggan,stylegan,gaugan,stylegan2,ddpm,adm,iddpm,dalle,glide,ldm,pndm,wukong,sdv1.4,midjourney,sdv1.5,vqdm,sdv2.1,sdxl1.0,sd-turbo,sdxl-turbo,sdv3.0,pixart,flux.1"
NAME="NPR"
CL_METHODS=("Seq" "ER" "EWC" "OSLA" "AGEM" "SI" "iCaRL" "Linear")
BS=256
####################################

cd ..

for CL_METHOD in "${CL_METHODS[@]}"; do
    echo "Running with CL method: $CL_METHOD"
    python -W ignore script/CL_test_AA_AF.py \
    --dataroot $DATAROOT \
    --dataset_name $DATASET_NAME \
    --multiclass 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
    --detection_model $NAME \
    --model_dir ./outputs/continual_learning \
    --output_dir ./CL_results \
    --checkpoint_name $CHECKPOINT_NAME \
    --cl_method $CL_METHOD \
    --batch_size $BS
done
