#!/bin/bash

export PYTHONPATH=..:$PYTHONPATH

TESTSETS=("ProGAN" "LSUN-bedroom" "GenImage" "cvpr23_ojha" "DRCT-2M" "DiTFake")
DEVICE="cuda:4"
NAME="GramNet"
MODEL_PATH="weights/classifier/GramNet/Gram.pth"
BS=32

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
    --batch_size "$BS"
done

echo "All tests completed."
