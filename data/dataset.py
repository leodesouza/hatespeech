import json
import os

from configs.database import database_path


# from abc
class DatasetLoader:
    def __init__(self):
        self.img_resized_files = []
        self.img_txt_files = []
        self.labels = []

    def load_dataset(self):

        self.load_labels()
        _labels = self.labels

        self.load_img_txt_files()
        texts = [item['img_text'] for item in self.img_txt_files]

        self.load_img_resized_files()
        images_path = self.img_resized_files

        return _labels, texts, images_path

    def load_img_txt_files(self):
        img_txt_path = database_path['img_txt_path']
        all_json_files = []
        for file_name in os.listdir(img_txt_path):
            if file_name.endswith('.json'):
                img_txt_file = os.path.join(img_txt_path, file_name)

            with open(img_txt_file, 'r') as file:
                data = json.load(file)
                all_json_files.append(data)

        self.img_txt_files = all_json_files

    def load_img_resized_files(self):
        img_resized_path = database_path['img_resized_path']
        # with open(database_path['database_path'])
        self.img_resized_files = []
        for file_name in os.listdir(img_resized_path):
            if file_name.endswith('.jpg'):
                img_resized_file_path = os.path.join(img_resized_path, file_name)
                self.img_resized_files.append(img_resized_file_path)

    def load_labels(self):
        # MMHS150K_GT
        dataset_file_path = database_path['MMHS150K_GT_json_file']
        dict_dataset = None
        with open(dataset_file_path, 'r') as file:
            dict_dataset = json.load(file)

        for key in dict_dataset:
            info = dict_dataset[key]
            _labels = info['labels']
            result_string = ','.join(map(str, _labels))
            self.labels.append(result_string)





