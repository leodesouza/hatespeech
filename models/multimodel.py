from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, concatenate, Conv2D, MaxPool2D, MaxPooling2D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.src import regularizers
from keras.src.applications import InceptionV3
from keras.src.initializers.initializers import HeNormal
from keras.src.layers import Dropout, BatchNormalization
from keras.src.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
import seaborn as sns

from configs.training_hyperparameters_config import config
from data.dataset import DatasetLoader
import matplotlib.pyplot as plt
import gensim.downloader as api
import numpy as np


def load_glove_embeddings(embedding_dim):
    glove_file_path = config['glove_file_100d']
    embedding_index = {}
    with open(glove_file_path, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index


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
        # self.w2v_model = api.load('word2vec-google-news-300')
        self.inception_model = InceptionV3(weights='imagenet',
                                           include_top=False,
                                           input_shape=(299, 299, 3))
        for layer in self.inception_model.layers:
            layer.trainable = False

    def create(self):
        # self.create_model2()
        self.create_model2()

    # def create_model1(self):
    #     tokenizer = self.dataset_loader.get_tokenized_tweet_texts()
    #     word_size = len(tokenizer.word_index) + 1
    #     text_input = Input(shape=(1,), dtype='int32', name='text_input')
    #     embedded_text = Embedding(input_dim=word_size,
    #                               output_dim=128,
    #                               input_length=self.max_sequence_length)(text_input)
    #     lstm_tweet_text = LSTM(128)(embedded_text)
    #
    #     image_input = Input(shape=(224, 224, 3), name='image_input')
    #     flattened_image = Flatten()(image_input)
    #     flattened_image = Dropout(0.2)(flattened_image)
    #
    #     merged = concatenate([lstm_tweet_text, flattened_image])
    #     merged = Dense(300, activation='relu')(merged)
    #     merged = BatchNormalization()(merged)
    #     merged = Dropout(0.3)(merged)
    #
    #     output = Dense(1, activation='sigmoid')(merged)
    #
    #     self.model = Model(inputs=[text_input, image_input], outputs=output)
    #     model_inputs = self.model.input
    #     for i, input_layer in enumerate(model_inputs):
    #         print(f"Input {i + 1}:")
    #         print(f"Name: {input_layer.name}")
    #         print(f"Shape: {input_layer.shape}")
    #         print(f"Dtype: {input_layer.dtype}")
    #         print(f"Data: {input_layer}")
    #         print()

    # def tokens_to_w2v_embeddings(self, tokens, embedding_dim):
    #     embedding_matrix = np.zeros((len(tokens), embedding_dim))
    #     for i, word in enumerate(tokens):
    #         if word in self.w2v_model:
    #             embedding_matrix[i] = self.w2v_model[word]
    #     return embedding_matrix

    def create_lstm_layer_tweet_text(self, text_input):
        word_index = self.dataset_loader.tokenizer.word_index
        sequences = self.dataset_loader.tweet_text_sequences
        max_sequence_length = max([len(seq) for seq in sequences])

        embedding_dim = 100
        embedding_index = load_glove_embeddings(embedding_dim)
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedded_text = Embedding(len(word_index) + 1,
                                  embedding_dim,
                                  weights=[embedding_matrix],
                                  input_length=max_sequence_length,
                                  trainable=False)(text_input)
        return LSTM(100)(embedded_text)

    def create_lstm_layer_image_text(self, text_input):
        tokenizer = self.dataset_loader.image_text_tokenizer
        word_size = len(tokenizer.word_index) + 1
        # embedding_dim = self.w2v_model.vector_size
        # embedding_matrix = self.tokens_to_w2v_embeddings(tokenizer.word_index, embedding_dim)
        embedded_text = Embedding(input_dim=word_size,
                                  output_dim=50,
                                  input_length=self.max_sequence_length)(text_input)
        return LSTM(50)(embedded_text)

    # def create_lstm_layer_image_text(self, text_input):
    #     tokenizer = self.dataset_loader.image_text_tokenizer
    #     word_size = len(tokenizer.word_index) + 1
    #     # embedding_dim = self.w2v_model.vector_size
    #     # embedding_matrix = self.tokens_to_w2v_embeddings(tokenizer.word_index, embedding_dim)
    #     embedded_text = Embedding(input_dim=word_size,
    #                               output_dim=50,
    #                               input_length=self.max_sequence_length)(text_input)
    #     return LSTM(150)(embedded_text)

    def create_model2(self):
        text_input = Input(shape=(1,), dtype='int32', name='text_input')
        lstm_text = self.create_lstm_layer_tweet_text(text_input)

        image_text_input = Input(shape=(1,), dtype='int32', name='image_text_input')
        lstm_image_text = self.create_lstm_layer_image_text(image_text_input)

        lstm_merged = concatenate([lstm_text, lstm_image_text])

        image_input = Input(shape=(299, 299, 3), name='image_input')
        image_features = self.inception_model(image_input)
        image_features = GlobalAveragePooling2D()(image_features)

        # kernel_regularizer = regularizers.l2(0.1)
        # kernel_initializer=HeNormal()
        merged = concatenate([lstm_merged, image_features])
        hidden1 = Dense(300, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(merged)

        hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.3)(hidden1)

        hidden2 = Dense(128, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(hidden1)
        hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.3)(hidden2)

        hidden3 = Dense(128, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(hidden2)
        hidden3 = BatchNormalization()(hidden3)
        hidden3 = Dropout(0.3)(hidden3)

        hidden4 = Dense(128, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(hidden3)
        hidden4 = BatchNormalization()(hidden4)
        hidden4 = Dropout(0.2)(hidden4)

        hidden5 = Dense(128, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(hidden4)
        hidden5 = BatchNormalization()(hidden5)
        hidden5 = Dropout(0.1)(hidden5)

        hidden6 = Dense(128, activation='relu',
                        kernel_initializer=HeNormal(),
                        kernel_regularizer=regularizers.l2(0.1))(hidden5)
        hidden6 = BatchNormalization()(hidden6)
        hidden6 = Dropout(0.1)(hidden6)

        # output = Dense(1, activation='sigmoid')(hidden2)
        output = Dense(6, activation='softmax')(hidden6)

        self.model = Model(inputs=[text_input, image_text_input, image_input], outputs=output)

        # model_inputs = self.model.input
        # for i, input_layer in enumerate(model_inputs):
        #     print(f"Input {i + 1}:")
        #     print(f"Name: {input_layer.name}")
        #     print(f"Shape: {input_layer.shape}")
        #     print(f"Dtype: {input_layer.dtype}")
        #     print(f"Data: {input_layer}")
        #     print()

    def create_model_3(self):
        text_input = Input(shape=(1,), dtype='int32', name='text_input')
        lstm_text = self.create_lstm_layer_tweet_text(text_input)

        image_text_input = Input(shape=(1,), dtype='int32', name='image_text_input')
        lstm_image_text = self.create_lstm_layer_image_text(image_text_input)

        image_input = Input(shape=(299, 299, 3), name='image_input')
        # image_features = self.inception_model(image_input)
        # image_features = GlobalAveragePooling2D()(image_features)

        conv_1 = Conv2D(32, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(image_input)
        pool_1 = MaxPooling2D((2, 2))(conv_1)

        conv_2 = Conv2D(50, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_1)
        pool_2 = MaxPooling2D((2, 2))(conv_2)

        conv_3 = Conv2D(30, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_2)
        pool_3 = MaxPooling2D((2, 2))(conv_3)
        global_pool = GlobalMaxPooling2D()(pool_3)

        merged = concatenate([lstm_text, lstm_image_text, global_pool])
        merged = Dense(40, activation='relu',
                       kernel_regularizer=regularizers.l2(0.1))(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.5)(merged)

        output = Dense(6, activation='softmax')(merged)

        self.model = Model(inputs=[text_input, image_text_input, image_input], outputs=output)

        # model_inputs = self.model.input
        # for i, input_layer in enumerate(model_inputs):
        #     print(f"Input {i + 1}:")
        #     print(f"Name: {input_layer.name}")
        #     print(f"Shape: {input_layer.shape}")
        #     print(f"Dtype: {input_layer.dtype}")
        #     print(f"Data: {input_layer}")
        #     print()

    def create_model4(self):
        text_input = Input(shape=(1,), dtype='int32', name='text_input')
        lstm_text = self.create_lstm_layer_tweet_text(text_input)

        image_text_input = Input(shape=(1,), dtype='int32', name='image_text_input')
        lstm_image_text = self.create_lstm_layer_image_text(image_text_input)

        image_input = Input(shape=(299, 299, 3), name='image_input')
        # image_features = self.inception_model(image_input)
        # image_features = GlobalAveragePooling2D()(image_features)

        conv_1 = Conv2D(32, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(image_input)
        pool_1 = MaxPooling2D((2, 2))(conv_1)

        conv_2 = Conv2D(50, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_1)
        pool_2 = MaxPooling2D((2, 2))(conv_2)

        conv_3 = Conv2D(30, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_2)
        pool_3 = MaxPooling2D((2, 2))(conv_3)

        conv_4 = Conv2D(30, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_3)
        pool_4 = MaxPooling2D((2, 2))(conv_4)

        conv_5 = Conv2D(30, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l2(0.001))(pool_4)
        pool_5 = MaxPooling2D((2, 2))(conv_5)

        global_pool = GlobalMaxPooling2D()(pool_5)

        merged = concatenate([lstm_text, lstm_image_text, global_pool])
        merged = Dense(40, activation='relu',
                       kernel_regularizer=regularizers.l2(0.1))(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.3)(merged)

        output = Dense(6, activation='softmax')(merged)

        self.model = Model(inputs=[text_input, image_text_input, image_input], outputs=output)

    def build(self):
        # weighted_metrics=[]
        # self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        text_data, text_image, image_data, labels = next(iterator)
        # text_data, image_data, labels = next(iterator)
        val_iterator = next(self.dataset_loader.val_hatespeech_dataset.as_numpy_iterator())
        val_text_data = val_iterator['text_data']
        val_image_text_data = val_iterator['image_text_data']
        val_image_data = val_iterator['image_data']
        val_labels_data = val_iterator['label_data']

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        training_history = self.model.fit(x=[text_data, text_image, image_data],
                                          y=labels,
                                          epochs=250,
                                          validation_data=(
                                              [val_text_data, val_image_text_data, val_image_data], val_labels_data),
                                          callbacks=[early_stopping])

        plot_training_result(training_history)

    def evaluate(self):
        iterator = self.dataset_loader.test_hatespeech_dataset.as_numpy_iterator()
        text_test_data, image_text, image_test_data, labels_test_data = next(iterator)

        results = self.model.evaluate([text_test_data, image_text, image_test_data], labels_test_data)

        print("Test loss: ", results[0])
        print("Test accuracy: ", results[1])
        self.predict(text_test_data, image_text, image_test_data, labels_test_data)

    def predict(self, test_text_data, test_image_text_data, test_imagem_data, labels_data):
        predictions = self.model.predict([test_text_data, test_image_text_data, test_imagem_data])
        predicted_labels = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(np.argmax(labels_data, axis=1), predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Não c.d ódio', 'Racista', 'Sexista', 'Homofóbico', 'Religioso', 'Outros'],
                    yticklabels=['Não c.d ódio', 'Racista', 'Sexista', 'Homofóbico', 'Religioso', 'Outros'])
        plt.xlabel('Saída Predita')
        plt.ylabel('Saída Real'),
        plt.title('Matriz de Confusão')
        plt.savefig('matriz_confusao.png')
