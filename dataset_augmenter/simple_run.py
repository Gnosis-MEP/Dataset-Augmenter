#!/usr/bin/env python
import os
import glob
from event_service_utils.streams.redis import RedisStreamFactory
from event_service_utils.img_serialization.redis import RedisImageCache

from dataset_augmenter.augmenter import Augmenter
from dataset_augmenter.disk_fs_cli import DiskImageLoader
from dataset_augmenter.annotators import ClassifierAnnotator

from dataset_augmenter.conf import (
    INPUT_BGS_DIR,
    INPUT_OIS_DIR,
    OUTPUT_DIR,
    OUTPUT_IMAGES_DIR_NAME,
    OUTPUT_ANNOTATION_NAME,
    REDIS_ADDRESS,
    REDIS_PORT,
    PUB_EVENT_LIST,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY_LIST,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
    SERVICE_DETAILS,
)


def get_bg_samples_uris(dataset_id):
    dataset_input_bgs_dir = os.path.join(INPUT_BGS_DIR, dataset_id)
    regexp = os.path.join(dataset_input_bgs_dir, '*.png')
    return glob.glob(regexp)


def get_oi_figs_uris_dict(oi_classes):
    oi_figs = {}
    for oi in oi_classes:
        dataset_input_oi_dir = os.path.join(INPUT_OIS_DIR, oi)
        # dict: str(OI label)->img_uri list
        regexp = os.path.join(dataset_input_oi_dir, '*.png')
        oi_figs[oi] = glob.glob(regexp)
    return oi_figs


def run_augmenter():

    dataset_id = 'TS-D-Q-1-5S'
    oi_classes = ['car']
    oi_delta = 7

    bg_samples = get_bg_samples_uris(dataset_id)
    bg_shape = (1920, 1080)

    all_classes = ['bg'] + oi_classes
    oi_fgs = get_oi_figs_uris_dict(oi_classes)
    oi_shape = (512, 512)

    fs_client = DiskImageLoader(
        output_dir=OUTPUT_DIR,
        json_annotation_name=OUTPUT_ANNOTATION_NAME,
        image_dir_name=OUTPUT_IMAGES_DIR_NAME,
        dataset_id=dataset_id
    )
    annotator=ClassifierAnnotator(
        classes=all_classes,

    )
    augmenter = Augmenter(
        annotator=annotator,
        fs_client=fs_client,
        dataset_id=dataset_id,
        bg_samples=bg_samples,
        bg_shape=bg_shape,
        oi_fgs=oi_fgs,
        oi_shape=oi_shape,
        oi_delta=oi_delta
    )
    augmenter.augment()


def main():
    try:
        run_augmenter()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()