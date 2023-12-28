from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, concatenate
from keras.src.layers import Dropout, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from data.dataset import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np


def plot_training_result(training_history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['loss'], label='Training Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Training and accuracy
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['accuracy'], label='Training Accuracy')
    plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_plot.png')
    # plt.show()


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
        embedded_text = Embedding(input_dim=word_size,
                                  output_dim=50,
                                  input_length=self.max_sequence_length)(text_input)
        lstm_text = LSTM(50)(embedded_text)

        # Image
        # image_input = Input(shape=(224, 224, 3), name='image_input')
        # flattened_image = Flatten()(image_input)
        #
        # merged = concatenate([lstm_text, flattened_image])
        # dense1 = Dense(128, activation='relu')(merged)
        # output = Dense(1, activation='softmax')(dense1)

        image_input = Input(shape=(224, 224, 3), name='image_input')
        flattened_image = Flatten()(image_input)
        flattened_image = Dropout(0.2)(flattened_image)

        merged = concatenate([lstm_text, flattened_image])
        merged = Dense(128, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.3)(merged)

        output = Dense(1, activation='sigmoid')(merged)

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
        # weighted_metrics=[]
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
        iterator = self.dataset_loader.train_hatespeech_dataset.as_numpy_iterator()
        text_data, image_data, labels = next(iterator)
        val_iterator = next(self.dataset_loader.val_hatespeech_dataset.as_numpy_iterator())
        # val_text_data, val_image_data, val_labels_data = next(val_iterator)
        val_text_data = val_iterator['text_data']
        val_image_data = val_iterator['image_data']
        val_labels_data = val_iterator['label_data']
        # val_labels_data = np.array(self.dataset_loader.labels_val)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        # callbacks=[early_stopping]
        # training_history = self.model.fit([text_data, image_data],
        #                                   labels,
        #                                   batch_size=31,
        #                                   epochs=10,
        #                                   validation_data=(val_text_data, val_image_data, val_labels_data),
        #                                   callbacks=[early_stopping])
        # plot_training_result(training_history)
        training_history = self.model.fit(x=[text_data, image_data],
                                          y=labels,
                                          epochs=250,
                                          validation_data=([val_text_data, val_image_data], val_labels_data),
                                          callbacks=[early_stopping])
        plot_training_result(training_history)

    def evaluate(self):
        iterator = self.dataset_loader.test_hatespeech_dataset.as_numpy_iterator()
        text_test_data, image_test_data, labels_test_data = next(iterator)

        results = self.model.evaluate([text_test_data, image_test_data], labels_test_data)
        print("Test loss: ", results[0])
        print("Test accuracy: ", results[1])
        pass
