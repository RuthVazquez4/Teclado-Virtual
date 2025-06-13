import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import pickle

# Crear carpeta model si no existe
if not os.path.exists("model"):
    os.makedirs("model")

# --- 1. Cargar palabras ---
with open("spanish_corpus.txt", encoding="utf-8") as f:
    words = f.read().splitlines()

chars = sorted(list(set("".join(words))))
char2idx = {c: i + 1 for i, c in enumerate(chars)}
idx2char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(char2idx) + 1

# --- 2. Crear secuencias ---
sequences = []
next_chars = []

for word in words:
    for i in range(1, len(word)):
        seq = word[:i]
        target = word[i]
        sequences.append([char2idx[c] for c in seq])
        next_chars.append(char2idx[target])

maxlen = max(len(seq) for seq in sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
y = to_categorical(next_chars, num_classes=vocab_size)

# --- 3. Modelo LSTM ---
model = Sequential([
    Embedding(vocab_size, 64, input_length=maxlen),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- 4. Entrenar ---
model.fit(X, y, epochs=30, batch_size=128)

# --- 5. Guardar modelo y mapas ---
model.save("model/autocomplete_lstm.h5")
with open("model/char_maps.pkl", "wb") as f:
    pickle.dump((char2idx, idx2char, maxlen), f)

print("Entrenamiento completado.")
