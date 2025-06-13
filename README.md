# Teclado Virtual

Este proyecto es un prototipo de teclado virtual controlado con gestos de la mano mediante visión por computadora. Además, incluye un sistema de autocompletado basado en un modelo LSTM entrenado con un corpus en español.

---

## 📋 Descripción

- Captura la mano frente a la cámara y detecta gestos para seleccionar teclas en un teclado virtual en pantalla.
- Permite escribir texto usando movimientos de dedos y seleccionar teclas visualmente.
- Incluye autocompletado de palabras usando un modelo LSTM entrenado con un corpus de palabras en español.
- Sonidos de clic para mejorar la experiencia de usuario.

---

## 🛠 Requisitos

- Python 3.8 o superior
- Cámara web funcional
- Librerías Python:
  - opencv-python
  - mediapipe
  - pyautogui
  - pygame
  - tensorflow

---

## Instalación

1. Clona o descarga este repositorio.

2. Instala las dependencias con pip:

   ```bash
   pip install opencv-python mediapipe pyautogui pygame tensorflow
