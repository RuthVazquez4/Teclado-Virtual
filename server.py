# --- Importar bibliotecas necesarias ---
import tensorflow as tf
import numpy as np
import pickle

# --- Cargar modelo LSTM entrenado y mapas de caracteres ---
# Se carga el modelo previamente guardado que predice el siguiente carácter en una secuencia
model = tf.keras.models.load_model('model/autocomplete_lstm.h5')

# Cargar los diccionarios de conversión y la longitud máxima de secuencia
with open('model/char_maps.pkl', 'rb') as f:
    char2idx, idx2char, maxlen = pickle.load(f)

# --- Función para predecir el siguiente carácter dado un texto parcial ---
def predict_next_char(text):
    text = text.lower()  # Convertir a minúsculas para mantener coherencia con el entrenamiento

    # Convertir cada carácter del texto a su índice correspondiente
    # Si el carácter no está en el diccionario, se usa 0 (padding)
    seq = [char2idx.get(c, 0) for c in text]

    # Aplicar padding para que la secuencia tenga longitud igual a maxlen
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=maxlen)

    # Realizar predicción con el modelo cargado
    preds = model.predict(seq_padded, verbose=0)[0]

    # Obtener el índice con la mayor probabilidad (carácter más probable)
    next_char_idx = np.argmax(preds)

    # Mapear el índice de vuelta a su carácter correspondiente
    next_char = idx2char.get(next_char_idx, '')

    return next_char

# --- Ejemplo interactivo de uso ---
# Permite al usuario ingresar texto parcial y obtener una sugerencia de siguiente carácter
if __name__ == "__main__":
    texto_entrada = input("Escribe el texto parcial: ")
    sugerencia = predict_next_char(texto_entrada)
    print(f"Siguiente caracter sugerido: '{sugerencia}'")
