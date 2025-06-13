import cv2               # Biblioteca para procesamiento de imágenes y video
import mediapipe as mp   # Biblioteca para detección y seguimiento de manos
import pyautogui         # Biblioteca para controlar el teclado y mouse del sistema
import time              # Biblioteca para manejo de tiempo y temporizadores
import math              # Biblioteca para funciones matemáticas (distancia)
import pygame            # Biblioteca para manejo de audio y multimedia
import re                # Biblioteca para expresiones regulares

# Inicializar pygame para reproducir sonido
pygame.init()
click_sound = pygame.mixer.Sound("click.wav")  # Cargar archivo de sonido para click

# Definir resolución deseada para captura de cámara
w, h = 1280, 720
cap = cv2.VideoCapture(0)    # Abrir cámara por defecto (índice 0)
cap.set(3, w)                # Establecer ancho del frame
cap.set(4, h)                # Establecer alto del frame

# Configurar MediaPipe Hands para detectar una mano con confianza mínima 0.8
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils  # Utilidad para dibujar puntos y conexiones

# Definición del teclado en filas con caracteres
keys = [
    list("1234567890"),           # Fila 1 con números
    list("QWERTYUIOP"),           # Fila 2 con letras superiores
    list("ASDFGHJKL"),            # Fila 3 con letras centrales
    list("ZXCVBNM"),              # Fila 4 con letras inferiores
    ['ESPACIO', 'BORRAR', 'COMPLETAR']  # Fila 5 con teclas especiales
]

# Tamaño y posición inicial para las teclas virtuales
key_w, key_h = 90, 90          # Ancho y alto de cada tecla
start_x, start_y = 100, 50     # Coordenadas iniciales para el teclado
x_spacing = 30                 # Espacio horizontal entre teclas
y_spacing = 30                 # Espacio vertical entre filas

# Variables para almacenar el texto ingresado y control de tiempos
texto_escrito = ""             # Texto que el usuario ha escrito
last_pressed = ""              # Última tecla detectada como presionada
last_time = 0                 # Tiempo de la última pulsación
click_threshold = 40           # Distancia mínima para detectar un click entre dedos

# --- Función para cargar vocabulario desde un archivo de texto ---
def cargar_vocabulario(path="spanish_corpus.txt", min_len=3, max_len=15):
    vocab = set()  # Usamos conjunto para evitar palabras repetidas
    try:
        with open(path, "r", encoding="utf-8") as f:
            for linea in f:
                # Extraemos palabras que contengan solo letras y caracteres especiales del español
                palabras = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', linea.lower())
                for p in palabras:
                    # Solo agregamos palabras que estén dentro del rango de longitud permitido
                    if min_len <= len(p) <= max_len:
                        vocab.add(p)
    except FileNotFoundError:
        print(f"Archivo {path} no encontrado. Usando vocabulario vacío.")
    return sorted(vocab)  # Devolvemos vocabulario ordenado

# Cargar el vocabulario para autocompletado
vocabulario = cargar_vocabulario()

# Variables para autocompletado y control de sugerencias
sugerencia = ""               # Palabra sugerida por autocompletado
ultimo_texto = ""             # Último texto escrito para evitar calcular sugerencias repetidas
last_pred_time = 0            # Última vez que se calculó sugerencia
pred_interval = 0.5           # Tiempo mínimo entre sugerencias (en segundos)

# --- Función simple de autocompletado ---
def autocomplete_simple(texto):
    palabras = texto.strip().lower().split()  # Dividir texto en palabras
    if not palabras:
        return ""  # Si no hay palabras, no sugerir nada
    ultima = palabras[-1]  # Tomar la última palabra escrita
    if ultima == "":
        return ""
    # Buscar en vocabulario palabras que empiecen igual pero no sean exactamente la misma
    posibles = [p for p in vocabulario if p.startswith(ultima) and p != ultima]
    if posibles:
        return posibles[0]  # Devolver la primera coincidencia
    return ""

# --- Función para dibujar el teclado virtual sobre la imagen ---
def draw_keyboard(img):
    key_positions = []  # Lista donde se guardan las posiciones y etiquetas de cada tecla
    y = start_y
    for row in keys:
        x = start_x
        for key in row:
            # Dibujar rectángulo relleno para tecla
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (200, 200, 200), -1)
            # Dibujar borde negro
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (0, 0, 0), 2)

            # Ajustar tamaño de texto según longitud de tecla (una letra o palabra)
            font_scale = 1.2 if len(key) == 1 else 0.8
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (key_w - text_size[0]) // 2
            text_y = y + (key_h + text_size[1]) // 2

            # Poner texto de la tecla en el centro
            cv2.putText(img, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

            # Guardar posición y etiqueta para detectar colisiones con el dedo
            key_positions.append((key, x, y))
            x += key_w + x_spacing
        y += key_h + y_spacing
    return key_positions

# --- Función para calcular distancia euclidiana entre dos puntos ---
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# --- Bucle principal para captura y procesamiento ---
while True:
    success, img = cap.read()       # Capturar frame de la cámara
    img = cv2.flip(img, 1)          # Voltear horizontal para espejo
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB para MediaPipe
    result = hands.process(rgb)     # Procesar imagen para detectar manos
    key_positions = draw_keyboard(img)  # Dibujar teclado en imagen actual

    current_time = time.time()      # Tiempo actual para controlar intervalos

    # Actualizar sugerencia solo si el texto ha cambiado y pasó suficiente tiempo
    if texto_escrito != ultimo_texto and (current_time - last_pred_time > pred_interval):
        sugerencia = autocomplete_simple(texto_escrito)
        ultimo_texto = texto_escrito
        last_pred_time = current_time

    # Si se detecta al menos una mano
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            # Obtener coordenadas de cada punto clave de la mano
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Dibujar puntos y conexiones de la mano sobre la imagen
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Si hay suficientes puntos (por seguridad)
            if len(lm_list) >= 9:
                index_finger = lm_list[8]  # Punta del dedo índice
                thumb_tip = lm_list[4]     # Punta del pulgar

                # Dibujar círculo en la punta del dedo índice
                cv2.circle(img, index_finger, 12, (255, 0, 255), -1)

                # Calcular distancia entre pulgar e índice para detectar gesto de clic
                d = distance(index_finger, thumb_tip)

                # Verificar si dedo índice está sobre alguna tecla
                for key, x, y in key_positions:
                    if x < index_finger[0] < x + key_w and y < index_finger[1] < y + key_h:
                        # Resaltar tecla seleccionada en verde
                        cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (0, 255, 0), -1)
                        cv2.putText(img, key, (x + 25, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                        # Si la distancia entre dedo índice y pulgar es menor que umbral y no es una pulsación repetida rápida
                        if d < click_threshold and (key != last_pressed or (current_time - last_time > 0.5)):
                            click_sound.play()  # Reproducir sonido de click

                            # Comportamiento según tecla presionada
                            if key == "ESPACIO":
                                pyautogui.press("space")   # Simular barra espaciadora
                                texto_escrito += " "
                            elif key == "BORRAR":
                                if texto_escrito:
                                    pyautogui.press("backspace")  # Simular retroceso
                                    texto_escrito = texto_escrito[:-1]
                            elif key == "COMPLETAR":
                                # Completar la palabra actual con la sugerencia
                                if sugerencia:
                                    palabras = texto_escrito.rstrip().split(" ")
                                    if palabras:
                                        ultima_palabra = palabras[-1]
                                        borrar_len = len(ultima_palabra)
                                        # Borrar la palabra incompleta con backspaces
                                        for _ in range(borrar_len):
                                            pyautogui.press("backspace")
                                        # Escribir la palabra sugerida completa
                                        pyautogui.write(sugerencia)
                                        palabras[-1] = sugerencia
                                        texto_escrito = " ".join(palabras)
                                        sugerencia = ""
                            else:
                                # Para teclas normales escribir letra en minúscula
                                pyautogui.write(key.lower())
                                texto_escrito += key.lower()

                            # Guardar tecla y tiempo de la pulsación actual
                            last_pressed = key
                            last_time = current_time

    # Dibujar cuadro blanco para mostrar texto escrito en la parte inferior
    cv2.rectangle(img, (50, h - 100), (w - 50, h - 40), (255, 255, 255), -1)
    # Escribir el texto acumulado
    cv2.putText(img, texto_escrito, (60, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Mostrar la sugerencia de autocompletado, si existe
    if sugerencia:
        cv2.putText(img, f"Sugerencia: {sugerencia}", (60, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)

    # Mostrar ventana con la imagen procesada y teclado virtual
    cv2.imshow("Teclado Virtual con Mano", img)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos y cerrar ventanas al finalizar
cap.release()
cv2.destroyAllWindows()
