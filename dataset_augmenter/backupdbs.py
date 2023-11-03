#!/usr/bin/env python
import json
import glob
import os
import sys
import shutil
import time

from dataset_augmenter.conf import OUTPUT_DIR, OUTPUT_IMAGES_DIR_NAME, OUTPUT_ANNOTATION_NAME, BACKUP_PROGRESS_JSON

DONE_IMAGES = []

if os.path.exists(BACKUP_PROGRESS_JSON):
    with open(BACKUP_PROGRESS_JSON, 'r') as f:
        DONE_IMAGES = json.load(f)
        print(len(DONE_IMAGES))


def slowly_copy_backup(dataset_id, bk_path, dry_run=False):
    aug_dataset_base_path = os.path.join(OUTPUT_DIR, dataset_id)

    bk_dataset_path = os.path.join(bk_path, dataset_id)

    if not os.path.exists(bk_dataset_path):
        os.mkdir(bk_dataset_path)

    annotation_file_path = os.path.join(aug_dataset_base_path, OUTPUT_ANNOTATION_NAME)
    annotation_new_path = os.path.join(bk_dataset_path, OUTPUT_ANNOTATION_NAME)
    print(f'annotation file')
    if dry_run:
        print(f'{annotation_file_path} -> {annotation_new_path}')
    else:
        shutil.copy(annotation_file_path, annotation_new_path)

    bk_images_path = os.path.join(bk_dataset_path, OUTPUT_IMAGES_DIR_NAME)
    if not os.path.exists(bk_images_path):
        os.mkdir(bk_images_path)
    images_regexp =  os.path.join(aug_dataset_base_path, OUTPUT_IMAGES_DIR_NAME, '*.png')

    image_paths = list(sorted(glob.glob(images_regexp)))
    total = len(image_paths)

    print(f'img files:')
    try:
        for i, img_path in enumerate(image_paths):
            image_name = os.path.basename(img_path)
            if image_name in DONE_IMAGES:
                continue
            new_img_path = os.path.join(bk_images_path, image_name)

            if dry_run:
                print(f'{img_path} -> {new_img_path}')
            else:
                shutil.copy(img_path, new_img_path)
            DONE_IMAGES.append(image_name)
            time.sleep(0.015)
            if i % 500 == 0:
                time.sleep(10)
                with open(BACKUP_PROGRESS_JSON, 'w') as f:
                    json.dump(DONE_IMAGES, f, indent=4)
            print(f'{i}/{total}:{i/total*100}')

    except Exception as e:
        print(e)
    finally:
        with open(BACKUP_PROGRESS_JSON, 'w') as f:
            json.dump(DONE_IMAGES, f, indent=4)





if __name__ == '__main__':
    dataset_id = sys.argv[1]
    bk_path = sys.argv[2]

    slowly_copy_backup(dataset_id, bk_path, dry_run=False)