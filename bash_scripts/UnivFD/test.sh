#!/bin/bash

export PYTHONPATH=..:$PYTHONPATH

TESTSETS=("ProGAN" "LSUN-bedroom" "GenImage" "cvpr23_ojha" "DRCT-2M" "DiTFake")
DEVICE="cuda:4"
NAME="UnivFD"
MODEL_PATH="weights/classifier/UnivFD/fc_weights.pth"
BS=4

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
    --arch 'CLIP:ViT-L/14'
done

echo "All tests completed."
