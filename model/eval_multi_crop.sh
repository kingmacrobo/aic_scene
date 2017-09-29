#!/usr/bin/env bash
BASE_DIR=..
LD_PRELOAD="/usr/lib/libtcmalloc.so.4" python eval_multi_crop.py --model_dir=${BASE_DIR}/checkpoints \
                --train_json=${BASE_DIR}/data/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json \
                --train_image_dir=${BASE_DIR}/data/ai_challenger_scene_train_20170904/scene_train_images_20170904 \
                --validate_json=${BASE_DIR}/data/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json \
                --validate_image_dir=${BASE_DIR}/data/ai_challenger_scene_validation_20170908/scene_validation_images_20170908 \
                --pretrained_model_path=${BASE_DIR}/pre_trained/inception_resnet_v2.ckpt
