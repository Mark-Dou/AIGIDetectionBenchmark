#!/bin/bash

export PYTHONPATH=..:$PYTHONPATH

TESTSETS=("ProGAN" "LSUN-bedroom" "GenImage" "cvpr23_ojha" "DRCT-2M" "DiTFake")
DEVICE="cuda:4"
NAME="FatFormer"
MODEL_PATH="weights/classifier/FatFormer/fatformer_4class_ckpt.pth"
BS=256

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
    --backbone 'CLIP:ViT-L/14' \
  --num_classes 2 \
  --num_vit_adapter 3 \
  --num_context_embedding 8
done

echo "All tests completed."
