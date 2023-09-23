import os


class SubDatasetCreation():
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def select_frames(self, initial_frames):
        pass

    def create_subset(self, subset_name, initial_frames):
        subset_path = os.path.join(self.output_path, subset_name)
        subset_frames_path = os.path.join(subset_path, 'train')
        frames_file_names = self.select_frames(initial_frames)
        os.mkdir(subset_path)
