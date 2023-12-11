import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json
import numpy as np

# Load JSON dataset
with open('your_dataset.json', 'r') as f:
    data = json.load(f)

# Extract text, image paths, and labels from the dataset
texts = [item['text'] for item in data]
image_paths = [item['image_path'] for item in data]
labels = [item['label'] for item in data]

# Tokenize and pad text data
max_words = 1000
max_sequence_length = 20

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
text_data = pad_sequences(sequences, maxlen=max_sequence_length)

# Load and preprocess image data
image_data = np.array([img_to_array(load_img(img, target_size=(224, 224))) for img in image_paths])

# Text input branch
text_input = Input(shape=(max_sequence_length,), dtype='int32')
embedded_text = Embedding(input_dim=max_words, output_dim=50, input_length=max_sequence_length)(text_input)
lstm_text = LSTM(50)(embedded_text)

# Image input branch
image_input = Input(shape=(224, 224, 3))
# Add your image processing layers here, such as convolutional layers, pooling, etc.
# For simplicity, we use a Flatten layer here.
flattened_image = Flatten()(image_input)

# Concatenate text and image branches
merged = concatenate([lstm_text, flattened_image])

# Fully connected layers for the combined model
dense1 = Dense(128, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense1)

# Create and compile the model
model = Model(inputs=[text_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your data
# model.fit([text_data, image_data], labels, epochs=...)
