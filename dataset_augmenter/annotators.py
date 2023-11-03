import json
import random

class BaseAnnotator():
    def __init__(self, classes):
        self.classes = classes
        self.non_oi_keys = []
        self.original_non_oi_keys = []
        self.has_oi_keys = []
        self.annotations = {
            'annotations': {}
        }

    def set_non_oi_keys(self, keys):
        self.non_oi_keys = keys
        self.original_non_oi_keys = [k for k in keys if k.endswith('_o')]

    def set_has_oi_keys(self, keys):
        self.has_oi_keys.extend(keys)

    def add_annotation(self, img_key, annotation):
        self.annotations['annotations'][img_key] = annotation

    def get_bg_sample_keys(self, bg_samples_len, original_only=True):
        if original_only:
            return self.original_non_oi_keys
        keys = self.non_oi_keys
        selected_bg_keys = random.sample(keys, bg_samples_len)

        return selected_bg_keys

    def create_base_example_annotation(self, class_id=None, objects=None):
        return {}

    def generate_annotation_data(self):
        return {
            'classes': {
                cls_i: { "name": cls }
                for cls_i, cls in enumerate(self.classes)
            },
            'annotations': self.annotations['annotations']
        }

class ClassifierAnnotator(BaseAnnotator):

    def create_base_example_annotation(self, class_id=None, objects=None):
        return {
            "category": class_id
        }
