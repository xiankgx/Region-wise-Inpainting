#!/bin/bash

python test.py \
    --test_data_path /home/gx/datasets/coco/test2017/ \
    --mask_path /home/gx/datasets/mask/testing_mask_dataset/ \
    --model_path continuous_places2/places2.ckpt-6666
