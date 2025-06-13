import cv2
import mediapipe as mp
import pyautogui
import time
import math
import pygame
import re

pygame.init()
click_sound = pygame.mixer.Sound("click.wav")

w, h = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

keys = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ['ESPACIO', 'BORRAR', 'COMPLETAR']
]

key_w, key_h = 90, 90
start_x, start_y = 100, 50
x_spacing = 30
y_spacing = 30

texto_escrito = ""
last_pressed = ""
last_time = 0
click_threshold = 40

# --- Cargar vocabulario desde spanish_corpus.txt ---
def cargar_vocabulario(path="spanish_corpus.txt", min_len=3, max_len=15):
    vocab = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for linea in f:
                # Limpiar línea y separar palabras
                palabras = re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+\b', linea.lower())
                for p in palabras:
                    if min_len <= len(p) <= max_len:
                        vocab.add(p)
    except FileNotFoundError:
        print(f"Archivo {path} no encontrado. Usando vocabulario vacío.")
    return sorted(vocab)

vocabulario = cargar_vocabulario()

sugerencia = ""
ultimo_texto = ""
last_pred_time = 0
pred_interval = 0.5

def autocomplete_simple(texto):
    palabras = texto.strip().lower().split()
    if not palabras:
        return ""
    ultima = palabras[-1]
    if ultima == "":
        return ""
    posibles = [p for p in vocabulario if p.startswith(ultima) and p != ultima]
    if posibles:
        return posibles[0]
    return ""

def draw_keyboard(img):
    key_positions = []
    y = start_y
    for row in keys:
        x = start_x
        for key in row:
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (200, 200, 200), -1)
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (0, 0, 0), 2)
            font_scale = 1.2 if len(key) == 1 else 0.8
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x + (key_w - text_size[0]) // 2
            text_y = y + (key_h + text_size[1]) // 2
            cv2.putText(img, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            key_positions.append((key, x, y))
            x += key_w + x_spacing
        y += key_h + y_spacing
    return key_positions

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    key_positions = draw_keyboard(img)

    current_time = time.time()

    if texto_escrito != ultimo_texto and (current_time - last_pred_time > pred_interval):
        sugerencia = autocomplete_simple(texto_escrito)
        ultimo_texto = texto_escrito
        last_pred_time = current_time

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(lm_list) >= 9:
                index_finger = lm_list[8]
                thumb_tip = lm_list[4]
                cv2.circle(img, index_finger, 12, (255, 0, 255), -1)
                d = distance(index_finger, thumb_tip)

                for key, x, y in key_positions:
                    if x < index_finger[0] < x + key_w and y < index_finger[1] < y + key_h:
                        cv2.rectangle(img, (x, y), (x + key_w, y + key_h), (0, 255, 0), -1)
                        cv2.putText(img, key, (x + 25, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

                        if d < click_threshold and (key != last_pressed or (current_time - last_time > 0.5)):
                            click_sound.play()
                            if key == "ESPACIO":
                                pyautogui.press("space")
                                texto_escrito += " "
                            elif key == "BORRAR":
                                if texto_escrito:
                                    pyautogui.press("backspace")
                                    texto_escrito = texto_escrito[:-1]
                            elif key == "COMPLETAR":
                                if sugerencia:
                                    palabras = texto_escrito.rstrip().split(" ")
                                    if palabras:
                                        ultima_palabra = palabras[-1]
                                        borrar_len = len(ultima_palabra)
                                        for _ in range(borrar_len):
                                            pyautogui.press("backspace")
                                        pyautogui.write(sugerencia)
                                        palabras[-1] = sugerencia
                                        texto_escrito = " ".join(palabras)
                                        sugerencia = ""
                            else:
                                pyautogui.write(key.lower())
                                texto_escrito += key.lower()

                            last_pressed = key
                            last_time = current_time

    cv2.rectangle(img, (50, h - 100), (w - 50, h - 40), (255, 255, 255), -1)
    cv2.putText(img, texto_escrito, (60, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    if sugerencia:
        cv2.putText(img, f"Sugerencia: {sugerencia}", (60, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)

    cv2.imshow("Teclado Virtual con Mano", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()