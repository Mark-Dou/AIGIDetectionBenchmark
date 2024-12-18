#!/bin/bash

export PYTHONPATH=..:$PYTHONPATH

TESTSETS=("ProGAN" "LSUN-bedroom" "GenImage" "cvpr23_ojha" "DiTFake" "DRCT-2M")
DEVICE="cuda:4"
NAME="SimE"
MODEL_PATH="weights/classifier/SimE/vit_l_14_model_epoch_best.pth"
BS=16

cd ..

for TESTSET in "${TESTSETS[@]}"; do
  echo "Testing on dataset: $TESTSET"
  python script/test.py \
    --test_dataset "$TESTSET" \
    --device "$DEVICE" \
    --results_dir "results/$NAME" \
    --name "$NAME" \
    --detection_model "$NAME" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BS" \
    --arch 'ViT-L/14'
done

echo "All tests completed."
