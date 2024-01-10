import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from configs.database import database_path
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

data_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1. / 255
)


def load_and_preprocess_image(image_path_file):
    img = tf.io.read_file(image_path_file)
    img = tf.image.decode_jpeg(img, channels=3)
    target_size = (299, 299)
    img = tf.image.resize(img, target_size)
    img /= 255.0  # normalize to [0,1]
    return img


class DatasetLoader:
    def __init__(self):
        self.tweet_text_sequences = None
        self.text_from_images_sequences = None
        self.image_text_tokenizer = None
        self.val_image_text_dataset = None
        self.val_ids_dataset = None
        self.test_image_text_dataset = None
        self.test_ids_dataset = None
        self.train_image_text_dataset = None
        self.train_ids_dataset = None
        self.ids_val = None
        self.ids_test = None
        self.ids_train = None
        self.image_text_val = None
        self.image_text_train = None
        self.image_text_test = None
        self.Ids = None
        self.image_text = None
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
        with open(dataset_file_path, 'r') as file:
            dict_dataset = json.load(file)
        for key in dict_dataset:
            info = dict_dataset[key]
            _labels = info['labels']
            result_string = ','.join(map(str, _labels))
            self.labels.append(result_string)

    def load_mmhs150k(self):
        # MMHS150K_GT
        dataset_file_path = database_path['MMHS150K_GT_json_file']
        image_text_path = database_path['img_txt_path']

        # self.img_resized_files = self.load_img_resized_files()
        self.tweet_text = []
        self.labels = []
        self.image_text = []
        self.Ids = []
        dict_dataset = None
        with open(dataset_file_path, 'r') as file:
            dict_dataset = json.load(file)
        zero_digits = [6] * 6
        img_resized_path = database_path['img_resized_path']
        for key in dict_dataset:
            self.Ids.append(key)
            info = dict_dataset[key]
            self.labels.append(info['labels'])
            text = info['tweet_text']
            self.tweet_text.append(text)
            img_text_file_path = os.path.join(image_text_path, f'{key}.json')
            if os.path.exists(img_text_file_path):
                with open(img_text_file_path, 'r') as file:
                    img_text_dict = json.load(file)
                    txt = img_text_dict['img_text']
                    self.image_text.append(txt)
            else:
                self.image_text.append('')

            img_resized_file_path = os.path.join(img_resized_path, f'{key}.jpg')
            self.img_resized_files.append(img_resized_file_path)

        one_hot_encoded = []
        num_classes = 6
        for labels in self.labels:
            one_hot = np.zeros(num_classes)
            for label in labels:
                one_hot[label] = 1
            one_hot_encoded.append(one_hot)

        binary_labels = np.array(one_hot_encoded)
        self.labels = binary_labels
        combined_data = list(zip(self.Ids, self.tweet_text, self.image_text, self.img_resized_files, self.labels))
        train_combined_data, combined_data_temp, = train_test_split(combined_data, test_size=0.4, random_state=42)
        test_combined_data, val_combined_data, = train_test_split(combined_data_temp, test_size=0.5, random_state=42)

        self.ids_train, self.tweet_text_train, self.image_text_train, self.img_resized_train, self.labels_train = zip(
            *train_combined_data)
        self.ids_test, self.tweet_text_test, self.image_text_test, self.img_resized_test, self.labels_test = zip(
            *test_combined_data)
        self.ids_val, self.tweet_text_val, self.image_text_val, self.img_resized_val, self.labels_val = zip(
            *val_combined_data)

        self.tweet_text_test = self.converto_to_tokenized_tweet_texts(self.tweet_text_test)
        self.tweet_text_val = self.converto_to_tokenized_tweet_texts(self.tweet_text_val)
        self.tweet_text_train = self.converto_to_tokenized_tweet_texts(self.tweet_text_train)

        self.image_text_test = self.converto_to_tokenized_image_texts(self.image_text_test)
        self.image_text_val = self.converto_to_tokenized_image_texts(self.image_text_val)
        self.image_text_train = self.converto_to_tokenized_image_texts(self.image_text_train)

        list_2 = list(self.img_resized_train)
        self.train_image_text_dataset = tf.data.Dataset.from_tensor_slices(self.image_text_train)
        self.train_text_dataset = tf.data.Dataset.from_tensor_slices(self.tweet_text_train)
        self.train_image_dataset = tf.data.Dataset.from_tensor_slices(list_2)
        self.train_image_dataset = self.train_image_dataset.map(load_and_preprocess_image)
        self.train_labels_dataset = tf.data.Dataset.from_tensor_slices(list(self.labels_train))

        # combine training datasets
        self.train_hatespeech_dataset = tf.data.Dataset.zip(
            (self.train_text_dataset, self.train_image_text_dataset, self.train_image_dataset,
             self.train_labels_dataset))
        batch_size = 128
        self.train_hatespeech_dataset = self.train_hatespeech_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # create a slice of the dataset to test
        # self.test_ids_dataset = tf.data.Dataset.from_tensor_slices(self.ids_test)
        self.test_image_text_dataset = tf.data.Dataset.from_tensor_slices(self.image_text_test)
        self.test_text_dataset = tf.data.Dataset.from_tensor_slices(self.tweet_text_test)
        self.test_image_dataset = tf.data.Dataset.from_tensor_slices(list(self.img_resized_test))
        self.test_image_dataset = self.test_image_dataset.map(load_and_preprocess_image)
        self.test_labels_dataset = tf.data.Dataset.from_tensor_slices(list(self.labels_test))

        # combine the test datasets
        self.test_hatespeech_dataset = tf.data.Dataset.zip(
            (self.test_text_dataset, self.test_image_text_dataset, self.test_image_dataset, self.test_labels_dataset))

        self.test_hatespeech_dataset = self.test_hatespeech_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # create a slice of the dataset to validation
        # self.val_ids_dataset = tf.data.Dataset.from_tensor_slices(self.ids_val)
        self.val_image_text_dataset = tf.data.Dataset.from_tensor_slices(self.image_text_val)
        self.val_text_dataset = tf.data.Dataset.from_tensor_slices(self.tweet_text_val)
        self.val_image_dataset = tf.data.Dataset.from_tensor_slices(list(self.img_resized_val))
        self.val_image_dataset = self.val_image_dataset.map(load_and_preprocess_image)
        self.val_labels_dataset = tf.data.Dataset.from_tensor_slices(list(self.labels_val))

        self.val_hatespeech_dataset = tf.data.Dataset.zip({
            'text_data': self.val_text_dataset,
            'image_text_data': self.val_image_text_dataset,
            'image_data': self.val_image_dataset,
            'label_data': self.val_labels_dataset})

        self.val_hatespeech_dataset = self.val_hatespeech_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def converto_to_tokenized_tweet_texts(self, text):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(text)
        sequences = self.tokenizer.texts_to_sequences(text)
        self.tweet_text_sequences = pad_sequences(sequences)
        return self.tweet_text_sequences

    def converto_to_tokenized_image_texts(self, text):
        self.image_text_tokenizer = Tokenizer()
        self.image_text_tokenizer.fit_on_texts(text)
        sequences = self.image_text_tokenizer.texts_to_sequences(text)
        self.text_from_images_sequences = pad_sequences(sequences)
        return self.text_from_images_sequences

    def get_tokenized_tweet_texts(self):
        if not self.tokenizer or not isinstance(self.tokenizer, Tokenizer):
            raise ValueError("Tokenizer is not ready yet")
        return self.tokenizer
