import concurrent.futures

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, concatenate
from keras.src.layers import Lambda
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from data.dataset import DatasetLoader
import tensorflow as tf
import numpy as np


class Hatespeech:

    def __init__(self):
        self.dataset_loader = DatasetLoader()
        self.model = Model()
        self.image_data = None
        self.text_data = None
        self.images_path = None
        self.texts = None
        self.labels = None
        self.max_words = 1000
        self.max_sequence_length = 20

    def create(self):

        tokenizer = self.dataset_loader.get_tokenized_tweet_texts()
        word_size = len(tokenizer.word_index) + 1
        text_input = Input(shape=(1,), dtype='int32', name='text_input')
        # float_conversion = Lambda(lambda x: tf.strings.to_number(x, out_type=tf.float32))(text_input)

        print("Text Input Shape:", text_input.shape)
        print("Text Input Data:", text_input)
        print()

        embedded_text = Embedding(input_dim=word_size,
                                  output_dim=50,
                                  input_length=self.max_sequence_length)(text_input)
        lstm_text = LSTM(50)(embedded_text)

        print("lstm_text Input Shape:", lstm_text.shape)
        print("lstm_text Input Data:", lstm_text)
        print()

        # Image
        image_input = Input(shape=(224, 224, 3), name='image_input')
        flattened_image = Flatten()(image_input)

        print("image_input Input Shape:", image_input.shape)
        print("image_input Input Data:", image_input)
        print()

        print("flattened_image Input Shape:", flattened_image.shape)
        print("flattened_image Input Data:", flattened_image)
        print()

        merged = concatenate([lstm_text, flattened_image])
        dense1 = Dense(50, activation='relu')(merged)
        output = Dense(1, activation='softmax')(dense1)

        self.model = Model(inputs=[text_input, image_input], outputs=output)
        model_inputs = self.model.input
        for i, input_layer in enumerate(model_inputs):
            print(f"Input {i + 1}:")
            print(f"Name: {input_layer.name}")
            print(f"Shape: {input_layer.shape}")
            print(f"Dtype: {input_layer.dtype}")
            print(f"Data: {input_layer}")
            print()

    def build(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, show_shapes=True, to_file='model.png', show_layer_names=True)

    def load_dataset(self):
        self.dataset_loader.load_mmhs150k()

    def set_training_parameters(self):
        pass

    def normalize(self):
        pass

    def train(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # self.model.fit(self.dataset_loader.train_hatespeech_dataset,
        #                validation_data=self.dataset_loader.test_hatespeech_dataset,
        #                epochs=100,
        #                verbose=1,
        #                callbacks=[early_stopping],
        #                batch_size=32)

        # with open('example.txt', 'w') as file:
        #     # Write some text to the file
        #     for batch in self.dataset_loader.train_hatespeech_dataset:
        #         for i, element in enumerate(batch):
        #             file.write(f"Element {i + 1} Shape: {element.shape}")
        # text_data = self.dataset_loader.train_hatespeech_dataset[0]
        # image_data = self.dataset_loader.train_hatespeech_dataset[1]
        # labels = self.dataset_loader.train_hatespeech_dataset[2]
        iterator = self.dataset_loader.train_hatespeech_dataset.as_numpy_iterator()
        text_data, image_data, labels = next(iterator)

        # with open('example.txt', 'w') as file:
        #     # Write some text to the file
        #     for batch in labels:
        #         for i, element in enumerate(batch):
        #             file.write(f"Element {i + 1} Shape: {element.shape}")

        self.model.fit([text_data, image_data], labels, batch_size=31, epochs=1)

    def evaluate(self):
        #   results = self.model.evaluate(self.dataset_loader.test_hatespeech_dataset)
        #   print("Test loss: ", results[0])
        #   print("Test accuracy: ", results[1])
        pass
