# Importar bibliotecas necesarias 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import pickle

# Crear carpeta 'model' si no existe 
# Esta carpeta se utilizará para guardar el modelo entrenado y los mapas de caracteres
if not os.path.exists("model"):
    os.makedirs("model")

# --- 1. Cargar palabras desde un corpus en español ---
# Se asume que cada línea del archivo 'spanish_corpus.txt' contiene una palabra
with open("spanish_corpus.txt", encoding="utf-8") as f:
    words = f.read().splitlines()

# --- Crear vocabulario de caracteres únicos ---
# Se extraen todos los caracteres presentes en las palabras del corpus
chars = sorted(list(set("".join(words))))  # Lista ordenada de caracteres únicos

# Crear mapas para convertir caracteres a índices y viceversa
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # Se suma 1 para reservar el índice 0 para padding
idx2char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(char2idx) + 1  # Se suma 1 por el índice 0 de padding

# --- 2. Crear secuencias de entrada y etiquetas ---
# Para cada palabra, se crean pares (secuencia parcial, siguiente carácter)
sequences = []
next_chars = []

for word in words:
    for i in range(1, len(word)):
        seq = word[:i]  # Subcadena desde el inicio hasta el carácter i-1
        target = word[i]  # Carácter siguiente a la secuencia
        sequences.append([char2idx[c] for c in seq])  # Convertir secuencia a índices
        next_chars.append(char2idx[target])  # Índice del carácter objetivo

# --- Padding de las secuencias ---
# Se rellenan con ceros al inicio para igualar todas las secuencias a la longitud máxima
maxlen = max(len(seq) for seq in sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)

# --- One-hot encoding de las etiquetas ---
# Las etiquetas (caracteres objetivo) se convierten a vectores one-hot
y = to_categorical(next_chars, num_classes=vocab_size)

# --- 3. Definir modelo LSTM para autocompletado de palabras ---
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),  # Embedding de caracteres
    LSTM(128),                                                             # Capa LSTM con 128 unidades
    Dense(vocab_size, activation='softmax')                               # Capa de salida con softmax
])

# --- Compilar modelo ---
# Se usa 'categorical_crossentropy' ya que se trata de un problema de clasificación multiclase
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Mostrar resumen del modelo
model.summary()

# --- 4. Entrenamiento del modelo ---
# Entrena el modelo durante 30 épocas con un tamaño de lote de 128
model.fit(X, y, epochs=30, batch_size=128)

# --- 5. Guardar el modelo y los mapas de caracteres ---
model.save("model/autocomplete_lstm.h5")  # Guardar modelo entrenado

# Guardar los diccionarios de conversión y longitud máxima de secuencia
with open("model/char_maps.pkl", "wb") as f:
    pickle.dump((char2idx, idx2char, maxlen), f)

# --- Fin del entrenamiento ---
print("Entrenamiento completado.")
