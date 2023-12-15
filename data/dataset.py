from configs.database import database_path
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DatasetLoader:
    def __init__(self):
        self.tokenizer = None
        self.labels_val = None
        self.tweet_text_val = None
        self.img_resized_val = None
        self.val_hatespeech_dataset = None
        self.val_labels_dataset = None
        self.val_text_dataset = None
        self.val_image_dataset = None
        self.test_hatespeech_dataset = None
        self.test_dataset = None
        self.test_labels_dataset = None
        self.test_text_dataset = None
        self.test_image_dataset = None
        self.train_labels_dataset = None
        self.train_text_dataset = None
        self.train_image_dataset = None
        self.labels_test = None
        self.labels_train = None
        self.tweet_text_test = None
        self.tweet_text_train = None
        self.img_resized_test = None
        self.img_resized_train = None
        self.validation_data = None
        self.train_data = None
        self.train_hatespeech_dataset = None
        self.img_resized_files = []
        self.tweet_text = []
        self.labels = []
        self.label_encoder = LabelEncoder()

    def load_dataset(self):

        self.load_labels()
        _labels = self.labels

        self.load_img_txt_files()
        texts = [item['img_text'] for item in self.tweet_text]

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

        self.tweet_text = all_json_files

    def load_img_resized_files(self):
        img_resized_path = database_path['img_resized_path']
        self.img_resized_files = []
        for file_name in os.listdir(img_resized_path):
            if file_name.endswith('.jpg'):
                img_resized_file_path = os.path.join(img_resized_path, file_name)
                self.img_resized_files.append(img_resized_file_path)
        return self.img_resized_files

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

    def load_and_preprocess_image(self, image_path_file):
        img = tf.io.read_file(image_path_file)
        img = tf.image.decode_jpeg(img, channels=3)
        target_size = (224, 224)
        img = tf.image.resize(img, target_size)
        img /= 255.0  # normalize to [0,1]
        return img

    def load_mmhs150k(self):
        # MMHS150K_GT
        dataset_file_path = database_path['MMHS150K_GT_json_file']
        # self.img_resized_files = self.load_img_resized_files()
        self.tweet_text = []
        self.labels = []
        dict_dataset = None
        with open(dataset_file_path, 'r') as file:
            dict_dataset = json.load(file)

        img_resized_path = database_path['img_resized_path']
        for key in dict_dataset:
            info = dict_dataset[key]
            _labels = info['labels']
            result_string = ','.join(map(str, _labels))
            self.labels.append(result_string)
            text = info['tweet_text']
            self.tweet_text.append(text)
            img_resized_file_path = os.path.join(img_resized_path, f'{key}.jpg')
            self.img_resized_files.append(img_resized_file_path)

        self.tweet_text = self.converto_to_tokenized_tweet_texts(self.tweet_text)

        self.labels = self.label_encoder.fit_transform(self.labels)
        # split the dataset to test and validation
        self.tweet_text_train, self.tweet_text_test = train_test_split(self.tweet_text, test_size=0.2, random_state=42)
        self.img_resized_train, self.img_resized_test = train_test_split(self.img_resized_files, test_size=0.2, random_state=42)
        self.labels_train, self.labels_test = train_test_split(self.labels, test_size=0.2, random_state=42)

        # create a slice of the dataset to train
        self.train_text_dataset = tf.data.Dataset.from_tensor_slices(self.tweet_text_train)
        self.train_image_dataset = tf.data.Dataset.from_tensor_slices(self.img_resized_train)
        self.train_image_dataset = self.train_image_dataset.map(self.load_and_preprocess_image)
        self.train_labels_dataset = tf.data.Dataset.from_tensor_slices(self.labels_train)

        # combine training datasets
        self.train_hatespeech_dataset = tf.data.Dataset.zip((self.train_text_dataset, self.train_image_dataset, self.train_labels_dataset))
        batch_size = 32
        self.train_hatespeech_dataset = self.train_hatespeech_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # create a slice of the dataset to test
        self.test_text_dataset = tf.data.Dataset.from_tensor_slices(self.tweet_text_test)
        self.test_image_dataset = tf.data.Dataset.from_tensor_slices(self.img_resized_test)
        self.test_image_dataset = self.test_image_dataset.map(self.load_and_preprocess_image)
        self.test_labels_dataset = tf.data.Dataset.from_tensor_slices(self.labels_test)
        # combine the test datasets
        self.test_hatespeech_dataset = tf.data.Dataset.zip((self.test_text_dataset, self.test_image_dataset,  self.test_labels_dataset))
        self.test_hatespeech_dataset = self.test_hatespeech_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def converto_to_tokenized_tweet_texts(self, tweet_text):
        self.tokenizer = Tokenizer(1000)
        self.tokenizer.fit_on_texts(tweet_text)
        sequences = self.tokenizer.texts_to_sequences(tweet_text)
        return pad_sequences(sequences)

    def get_tokenized_tweet_texts(self):
        if not self.tokenizer or not isinstance(self.tokenizer, Tokenizer):
            raise ValueError("Tokenizer is not ready yet")
        return self.tokenizer
