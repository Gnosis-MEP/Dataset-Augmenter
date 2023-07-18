import random
import math
from itertools import combinations

class Augmenter(object):

    def __init__(self, fs_client, dataset_id, bg_samples, bg_shape, oi_fgs, oi_shape,  oi_delta=7):
        """
        fs_client: a file storage client, need to have methods:
            get_image_ndarray_by_key_and_shape(self, img_key, nd_shape),
            upload_inmemory_to_storage(self, img_numpy_array)
        dataset_id: just an id for this dataset, probably based on pub_id
        bg_samples: img_uri list
        oi_fgs: dict: str(OI label)->img_uri list
        oi_delta: int
        """
        self.NON_OI_LABEL = "non_oi"
        self.HAS_OI_LABEL = "has_oi"
        self.fs_client = fs_client
        self.dataset_id = dataset_id
        self.bg_samples = bg_samples
        self.bg_shape = bg_shape
        self.bg_samples_len = len(bg_samples)
        self.oi_delta = oi_delta
        self.oi_fgs = oi_fgs
        self.oi_shape = oi_shape
        self.oi_len = len(self.oi_fgs.keys())
        self.oi_comb_list = []
        self.oi_comb_list_len = 0
        self._setup_oi_combinations()
        self.class_size = self._calculate_class_size()
        self.bg_variations_size = 0
        self.bg_random_noise_size = 0
        self._setup_bg_variations()
        self.annotations = {
            'non_oi_keys': [],
            'has_oi_keys': [],
            'data': []
        }

    def _calculate_class_size(self):
        return self.bg_samples_len * self.oi_comb_list_len * self.oi_delta

    def _setup_oi_combinations(self):
        self.oi_comb_list = []
        for i in range(1, self.oi_len + 1):
            self.oi_comb_list.extend(combinations(self.oi_fgs.keys(), i))
        self.oi_comb_list_len = len(self.oi_comb_list)

    def _setup_bg_variations(self):
        self.bg_variations_size = max(3, math.ceil(self.class_size / self.bg_samples_len))
        self.bg_random_noise_size = self.bg_variations_size - 2


    def save_example(self, image_ndarray, annotation, ref=None):
        img_key = self.fs_client.upload_inmemory_to_storage(image_ndarray)
        # annotation = self.create_example_annotation(img_key)
        annotation['img_key'] = img_key
        self.annotations['data'].append(annotation)
        return img_key

    def create_base_example_annotation(self, class_label, objects):
        """
        class_label: non_oi, has_oi
        objects: dict-> key: object class label, value: list of bboxes
        """
        return {
            'class_label': class_label,
        }

    def bg_sample_origin(self, bg_image):
        origin_annotation = self.create_base_example_annotation(class_label=self.NON_OI_LABEL)
        return self.save_example(bg_image, origin_annotation, ref='o')

    def bg_sample_transform_denoise(self, bg_image):
        annotation = self.create_base_example_annotation(class_label=self.NON_OI_LABEL)
        transform_img = bg_image
        return self.save_example(transform_img, annotation, ref='dn')

    def bg_sample_transform_random_noise(self, bg_image, rn_delta):
        annotation = self.create_base_example_annotation(class_label=self.NON_OI_LABEL)
        transform_img = bg_image
        return self.save_example(transform_img, annotation, ref=f'rn_{rn_delta}')

    def bg_sample_augmentations(self, bg_image):
        augmented_examples = []

        augmented_examples.append(self.bg_sample_origin(bg_image))
        augmented_examples.append(self.bg_sample_transform_denoise(bg_image))
        for rn_i in range(self.bg_random_noise_size):
            augmented_examples.append(self.bg_sample_transform_random_noise(bg_image, rn_i))

        return augmented_examples

    def prepare_non_oi_examples(self):
        examples = []
        for bg_img_uri in self.bg_samples:
            image_ndarray = self.fs_client.get_image_ndarray_by_key_and_shape(bg_img_uri, self.bg_shape)
            augmented_examples = self.bg_sample_augmentations(image_ndarray)
            examples.extend(augmented_examples)
        return examples

    def select_fg_imgs_for_oi_comb(self, oi_comb):
        selected_oi_fg_imgs = {}
        for oi in oi_comb:
            n_samples = random.randint(1, len(self.oi_fgs[oi]))
            oi_sample = random.sample(self.oi_fgs[oi], n_samples)
            samples_dict = {}
            for oi_img_uri in oi_sample:
                samples_dict[oi_img_uri] = self.fs_client.get_image_ndarray_by_key_and_shape(oi_img_uri, self.oi_shape)
            selected_oi_fg_imgs[oi] = samples_dict
        return selected_oi_fg_imgs

    def randomize_oi_fg_transform(self, oi_image):
        bbox = [1, 2, 3, 4]
        transformed_image = oi_image
        return transformed_image, bbox

    def prepare_fg_oi_img_combination(self, oi_images):
        "oi_images: dict-> key: oi label, values: dict (img_key: img)"
        objects = {}
        fg_image = None #nparray? cv2? blank image
        for oi, oi_data in oi_images.items():
            bbox_list = []
            for oi_image in oi_data.values():
                t_oi_image, bbox = self.randomize_oi_fg_transform(oi_image)
                bbox_list.append(bbox)
                fg_image += t_oi_image # not really but instead should concat each image into the final foreground based
            objects[oi] = bbox_list
        return fg_image, objects

    def oi_fg_augmentations(self, comb_i, bg_image, oi_images):
        augmented_examples = []
        for delta_i in range(self.oi_delta):
            fg_image, objects = self.prepare_fg_oi_img_combination(oi_images)
            final_image = bg_image + fg_image # not really, probably a concat with mask on white pixels
            annotation = self.create_base_example_annotation(class_label=self.HAS_OI_LABEL, objects=objects)
            augmented_examples.append(self.save_example(final_image, annotation, ref=f'c_{comb_i}_{delta_i}'))
        return augmented_examples

    def prepare_has_oi_examples(self):
        examples = []

        for comb_i, oi_comb in enumerate(self.oi_comb_list):
            oi_images = self.select_fg_imgs_for_oi_comb(oi_comb)
            # oi_images = [
            #     self.fs_client.get_image_ndarray_by_key_and_shape(oi_img_uri, self.oi_shape)
            #     for oi_img_uri in selected_oi_fg_keys
            # ]
            selected_bg_keys = random.sample(self.annotations['non_oi_keys'], self.bg_samples_len)
            for bg_key in selected_bg_keys:
                bg_image = self.fs_client.get_image_ndarray_by_key_and_shape(bg_key, self.bg_shape)
                augmented_examples = self.oi_fg_augmentations(comb_i, bg_image, oi_images)
                examples.extend(augmented_examples)


        # for bg_key in selected_bg_keys:
        #     image_ndarray = self.fs_client.get_image_ndarray_by_key_and_shape(bg_img_uri, self.bg_shape)
        #     augmented_examples = self.bg_sample_augmentations(image_ndarray)
        #     examples.extend(augmented_examples)
        return examples

    def _prepare_disk(self):
        "only for inital results, probably, maybe will need this on minio for the fs"
        # create directory in "./data/dataset_id"

    def augment(self):
        self._prepare_disk()
        self.annotations['non_oi_keys'] = self.prepare_non_oi_examples()
        self.annotations['has_oi_keys'] = self.prepare_has_oi_examples()