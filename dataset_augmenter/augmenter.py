import random
import math
from itertools import combinations
import numpy as np
import cv2

from .composition_utils import composite, composite_from_bbox, noisy

from .conf import (
    OUTPUT_DIR,
    OUTPUT_IMAGES_DIR_NAME,
    OUTPUT_ANNOTATION_NAME,
)

class Augmenter(object):

    def __init__(self, annotator, fs_client, dataset_id, bg_samples, bg_shape, oi_fgs, oi_shape,  oi_delta=7):
        """
        fs_client: a file storage client, need to have methods:
            get_image_ndarray_by_key_and_shape(self, img_key, nd_shape),
            upload_inmemory_to_storage(self, img_numpy_array)
        dataset_id: just an id for this dataset, probably based on pub_id
        bg_samples: img_uri list
        bg_shape: shape of the bg, eg: (1920, 180)
        oi_fgs: dict: str(OI label)->img_uri list
        oi_shape: shape of the object of interest, eg: (512, 512)
        oi_delta: int, number of FG variations for each OI class combination
        """

        self.NON_OI_ID = 0
        self.HAS_OI_ID = 1
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
        self.annotator = annotator
        # self.annotations = {
        #     'non_oi_keys': [],
        #     'has_oi_keys': [],
        #     'data': []
        # }

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


    def save_example(self, base_img_uri, image_ndarray, annotation, ref=None):
        img_key = self.fs_client.upload_inmemory_to_storage(base_img_uri, image_ndarray, ref=ref)
        # annotation = self.create_example_annotation(img_key)
        self.annotator.add_annotation(img_key, annotation)
        # annotation['img_key'] = img_key
        # self.annotations['data'].append(annotation)
        return img_key

    def create_base_example_annotation(self, class_label, objects):
        """
        class_label: non_oi, has_oi
        objects: dict-> key: object class label, value: list of bboxes
        """
        return {
            'class_label': class_label,
        }

    def bg_sample_origin(self, base_img_uri, bg_image):
        origin_annotation = self.annotator.create_base_example_annotation(class_id=self.NON_OI_ID)
        return self.save_example(base_img_uri, bg_image, origin_annotation, ref='o')

    def bg_sample_transform_denoise(self, base_img_uri, bg_image):
        # https://machinelearningprojects.net/blurrings-in-cv2/
        origin_annotation = self.annotator.create_base_example_annotation(class_id=self.NON_OI_ID)

        # transform_img = cv2.fastNlMeansDenoisingColored(bg_image, None, 10, 10, 7, 21)
        transform_img = cv2.medianBlur(bg_image, ksize=5)
        return self.save_example(base_img_uri, transform_img, origin_annotation, ref='dn')

    def bg_sample_transform_random_noise(self, base_img_uri, bg_image, rn_delta):
        # https://medium.com/mlearning-ai/how-to-denoise-an-image-using-median-blur-in-python-using-opencv-easy-project-50c2de13ac33
        origin_annotation = self.annotator.create_base_example_annotation(class_id=self.NON_OI_ID)

        # p = 0.2
        # noisy = np.zeros(bg_image.shape, np.uint8)
        # #traversing throughout the image pixels
        # for i in range(bg_image.shape[0]): #rows
        #     for j in range(bg_image.shape[1]): #cols
        #         r = random.random()
        #         if r < p / 2:
        #             noisy[i][j] = [0, 0, 0] #black noise
        #         elif r < p:
        #             noisy[i][j] = [255, 255, 255] #white noise
        #         else:
        #             noisy[i][j] = bg_image[i][j] #original image pixel
        transform_img = noisy('gauss', bg_image)
        return self.save_example(base_img_uri, transform_img, origin_annotation, ref=f'rn_{rn_delta}')

    def bg_sample_augmentations(self, img_uri, bg_image):
        augmented_examples = []
        augmented_examples.append(self.bg_sample_origin(img_uri, bg_image))
        augmented_examples.append(self.bg_sample_transform_denoise(img_uri, bg_image))
        for rn_i in range(self.bg_random_noise_size):
            augmented_examples.append(self.bg_sample_transform_random_noise(img_uri, bg_image, rn_i))

        return augmented_examples

    def prepare_non_oi_examples(self):
        examples = []
        for bg_img_uri in self.bg_samples:
            image_ndarray = self.fs_client.get_image_ndarray_by_key_and_shape(bg_img_uri, self.bg_shape)
            augmented_examples = self.bg_sample_augmentations(bg_img_uri, image_ndarray)
            examples.extend(augmented_examples)

            # remove this (add to reduce exampels and test)
            break
        return examples

    def crop_oi_fig(self, image_ndarray):
        # im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Set a threshold for what you consider "close to zero"
        threshold = 30  # Adjust this threshold as needed

        # Get the coordinates where alpha values are not close to zero
        y, x = np.where(np.abs(image_ndarray[:, :, 3] - 0) > threshold)

        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        cropImg = image_ndarray[miny:maxy, minx:maxx]
        return cropImg

    def select_fg_imgs_for_oi_comb(self, oi_comb):
        selected_oi_fg_imgs = {}
        for oi in oi_comb:
            n_samples = random.randint(1, len(self.oi_fgs[oi]))
            oi_sample = random.sample(self.oi_fgs[oi], n_samples)

            samples_dict = {}
            for oi_img_uri in oi_sample:
                samples_dict[oi_img_uri] = self.crop_oi_fig(
                    self.fs_client.get_image_ndarray_by_key_and_shape(oi_img_uri, self.oi_shape, alpha=True)
                )
            selected_oi_fg_imgs[oi] = samples_dict
        return selected_oi_fg_imgs

    def randomize_oi_fg_transform(self, fg_size, oi_image):
        rand_resize = random.randint(50, 150) / 100
        new_height = int(oi_image.shape[0] * rand_resize)
        new_width = int(oi_image.shape[1] * rand_resize)

        x1, y1 = [random.random(), random.random()]
        x2 = min(1.0, x1 + (new_width /fg_size[0]))
        y2 = min(1.0, y1 + (new_height /fg_size[1]))
        # x2 = x1 + (new_width /fg_size[0])
        # y2 = y1 + (new_height /fg_size[1])

        resize_interp = cv2.INTER_LINEAR
        if rand_resize < 0:
            resize_interp = cv2.INTER_ARE
        resized_image = cv2.resize(oi_image, (new_width, new_height), interpolation=resize_interp)

        # flip_rand = random.choice(['v', 'h', 'n'])
        # fliped_image = resized_image
        # if flip_rand != 'n':
        #     if flip_rand == 'v':
        #         fliped_image = cv2.flip(fliped_image, 0)
        #     else:
        #         fliped_image = cv2.flip(fliped_image, 1)

        flip_rand = random.choice([True, False])
        fliped_image = resized_image
        if flip_rand:
            fliped_image = cv2.flip(fliped_image, 1)


        bbox = [x1, y1, x2, y2]
        transformed_image = fliped_image
        return transformed_image, bbox

    def prepare_fg_oi_img_combination(self, oi_images, fg_size):
        "oi_images: dict-> key: oi label, values: dict (img_key: img)"
        objects = {}
        fg_height, fg_width = fg_size
        # blank fg image
        fg_image = np.zeros((fg_height, fg_width, 4), np.uint8)
        for oi, oi_data in oi_images.items():
            # print(f"fg images: {oi_data.keys()}")
            bbox_list = []
            for oi_image in oi_data.values():
                t_oi_image, bbox = self.randomize_oi_fg_transform(fg_size, oi_image)
                # print(f'> {bbox}')
                fg_image = composite_from_bbox(src=t_oi_image, dst=fg_image, bbox=bbox)
                bbox_list.append(bbox)
            objects[oi] = bbox_list
            # if any([i > 0.9 for i in bbox[:2]]):
            #     import ipdb; ipdb.set_trace()
            #     cv2.imwrite('teste-r.png', fg_image)
            #     pass
        return fg_image, objects

    def oi_fg_augmentations(self, base_img_uri, comb_i, bg_image, oi_images):
        augmented_examples = []
        fg_size = bg_image.shape[:2]

        for delta_i in range(self.oi_delta):
            # print(f'preparing for {base_img_uri} c_{comb_i}_{delta_i}')
            fg_image, objects = self.prepare_fg_oi_img_combination(oi_images, fg_size)
            annotation = self.annotator.create_base_example_annotation(class_id=self.HAS_OI_ID, objects=objects)
            self.save_example(base_img_uri, fg_image, annotation, ref=f'c_{comb_i}_{delta_i}_fg')
            final_image = composite(src=fg_image, dst=bg_image.copy())
            augmented_examples.append(self.save_example(base_img_uri, final_image, annotation, ref=f'c_{comb_i}_{delta_i}'))
            # todo: remove the break
            break

        return augmented_examples

    def prepare_has_oi_examples(self):
        examples = []

        for comb_i, oi_comb in enumerate(self.oi_comb_list):
            oi_images = self.select_fg_imgs_for_oi_comb(oi_comb)
            # oi_images = [
            #     self.fs_client.get_image_ndarray_by_key_and_shape(oi_img_uri, self.oi_shape)
            #     for oi_img_uri in selected_oi_fg_keys
            # ]

            #use this instead!:
            self.bg_samples_len = 3 # remove this line!
            selected_bg_keys = self.annotator.get_bg_sample_keys(self.bg_samples_len)

            # selected_bg_keys = random.sample(self.annotations['non_oi_keys'], self.bg_samples_len)
            for bg_key in selected_bg_keys:
                bg_image = self.fs_client.get_image_ndarray_by_key_and_shape(bg_key, self.bg_shape)
                augmented_examples = self.oi_fg_augmentations(bg_key, comb_i, bg_image, oi_images)
                examples.extend(augmented_examples)

            # remove this (add to reduce exampels and test)
            break



        # for bg_key in selected_bg_keys:
        #     image_ndarray = self.fs_client.get_image_ndarray_by_key_and_shape(bg_img_uri, self.bg_shape)
        #     augmented_examples = self.bg_sample_augmentations(image_ndarray)
        #     examples.extend(augmented_examples)
        return examples

    def _prepare_disk(self):
        "only for inital results, probably, maybe will need this on minio for the fs"
        # create directory in "./data/dataset_id"
        self.fs_client.prepare_disk()

    def save_annotations(self):
        self.fs_client.upload_annotation_inmemory_to_storage(self.annotator.generate_annotation_data())


    def augment(self):
        self._prepare_disk()
        self.annotator.non_oi_keys = self.prepare_non_oi_examples()
        self.annotator.has_oi_keys = self.prepare_has_oi_examples()
        self.save_annotations()

def main():
    from dataset_augmenter.disk_fs_cli import DiskImageLoader
    fs_cli = DiskImageLoader()
    dataset_id = 'EarlyResults'
    bg_samples = {}
    bg_shape = (100, 100)
    oi_figs = {}
    oi_shape = [(100, 100)]
    oi_delta = 3
    augmenter = Augmenter(fs_cli, dataset_id, bg_samples, bg_shape, oi_figs, oi_shape, oi_delta)
    augmenter.augment()

if __name__ == '__main__':
    main()
