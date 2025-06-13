import tensorflow as tf
import numpy as np
import pickle

# Cargar modelo y diccionarios
model = tf.keras.models.load_model('model/autocomplete_lstm.h5')
with open('model/char_maps.pkl', 'rb') as f:
    char2idx, idx2char, maxlen = pickle.load(f)

def predict_next_char(text):
    text = text.lower()
    # Convertir texto a indices usando char2idx, ignorar caracteres desconocidos
    seq = [char2idx.get(c, 0) for c in text]
    # Padding
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=maxlen)
    
    # Predecir
    preds = model.predict(seq_padded, verbose=0)[0]
    next_char_idx = np.argmax(preds)
    
    # Mapear índice a char
    next_char = idx2char.get(next_char_idx, '')
    return next_char

# Ejemplo de uso
if __name__ == "__main__":
    texto_entrada = input("Escribe el texto parcial: ")
    sugerencia = predict_next_char(texto_entrada)
    print(f"Siguiente caracter sugerido: '{sugerencia}'")