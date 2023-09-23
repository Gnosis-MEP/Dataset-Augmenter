import json
import os

import cv2

class DiskImageLoader():
    def __init__(self, output_dir, json_annotation_name, image_dir_name, dataset_id):
        self.output_dir = output_dir
        self.image_dir_name = image_dir_name
        self.json_annotation_name = json_annotation_name
        self.aug_dataset_base_path = os.path.join(self.output_dir, dataset_id)
        self.aug_dataset_image_path = os.path.join(self.aug_dataset_base_path, self.image_dir_name)
        self.aug_dataset_json_path = os.path.join(self.aug_dataset_base_path, json_annotation_name)

    def prepare_disk(self):
        if not os.path.exists(self.aug_dataset_base_path):
            os.mkdir(self.aug_dataset_base_path)
        if not os.path.exists(self.aug_dataset_image_path):
            os.mkdir(self.aug_dataset_image_path)

    def get_image_ndarray_by_key_and_shape(self, img_uri, shape, alpha=False):
        if '/' not in img_uri:
            img_uri = os.path.join(self.aug_dataset_image_path, f'{img_uri}.png')

        if alpha:
            return cv2.imread(img_uri, cv2.IMREAD_UNCHANGED)
        return cv2.imread(img_uri)

    def upload_inmemory_to_storage(self, base_image_uri, img_numpy_array, ref=None):
        base_image_id = os.path.basename(base_image_uri).split('.')[0]
        new_img_key = f'{base_image_id}_{ref}'
        img_path = os.path.join(self.aug_dataset_image_path, f'{new_img_key}.png')
        cv2.imwrite(img_path, img_numpy_array)
        return new_img_key

    def upload_annotation_inmemory_to_storage(self, annotation_data):
        with open(self.aug_dataset_json_path, 'w') as f:
            json.dump(annotation_data, f)
