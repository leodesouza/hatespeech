from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, concatenate
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences, img_to_array, load_img
from data.dataset import DatasetLoader
import numpy as np

import tensorflow as tf


class Hatespeech:

    def __init__(self):
        self.model = Model()
        self.image_data = None
        self.text_data = None
        self.images_path = None
        self.texts = None
        self.labels = None
        self.max_words = 1000
        self.max_sequence_length = 20

    def build(self):
        # Text input branch
        text_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_text = Embedding(input_dim=self.max_words, output_dim=50, input_length=self.max_sequence_length)(
            text_input)
        lstm_text = LSTM(50)(embedded_text)

        # Image input branch
        image_input = Input(Shape=(224, 224, 3))
        flattened_image = Flatten()(image_input)
        merged = concatenate([lstm_text, flattened_image])
        dense1 = Dense(128, activation='relu')(merged)
        output = Dense(1, activation='sigmoid')(dense1)

        self.model = Model(inputs=[text_input, image_input], outputs=output)

    def load_dataset(self):
        loader = DatasetLoader()
        _labels, _texts, _images_path = loader.load_dataset()
        self.labels = _labels
        self.texts = _texts
        self.images_path = _images_path

    def pre_processdata(self):
        self.preprocess_text()
        self.preprocess_images()

    def preprocess_text(self):
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(self.texts)
        sequences = tokenizer.texts_to_sequences(self.texts)
        self.text_data = pad_sequences(sequences, maxlen=self.max_sequence_length)

    def preprocess_images(self):
        np_array = []
        for img_file in self.images_path:
            _img = load_img(img_file, target_size=(898, 500))
            np_array.append(img_to_array(_img))
        self.image_data = np.array(np_array)
        j = 10

    def set_training_parameters(self):
        pass

    def normalize(self):
        pass

    def train(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def evaluate(self):
        pass
