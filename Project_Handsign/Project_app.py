import tkinter as tk
from tkinter import filedialog, LabelFrame
import cv2
from PIL import Image, ImageTk
import cv2
import os
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os
import pygame
import threading
from gtts import gTTS
import io

root = tk.Tk()
root.title("Video Processing App")
root.geometry("1000x800")

root.columnconfigure((0, 1), weight=1)
root.rowconfigure((0, 1), weight=1)

cap = None


def open_app1_BCC():
    subprocess.Popen(["python", "app_BCC.py"])

def clear_text():
    global result_text
    result_text = ""  
    text_box.configure(state="normal")   
    text_box.delete("1.0", tk.END)
    text_box.configure(state="disabled")

def save_text():
    content = text_box.get("1.0", tk.END)
    with open("saved_text.txt", "w", encoding="utf-8") as file:
        file.write(content)

def open_camera():
    global cap, running
    running = True
    cap = cv2.VideoCapture(0)
    process_video_with_prediction()  

def open_video():
    global cap, running
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        running = True
        cap = cv2.VideoCapture(file_path)
        process_video_with_prediction()  
        

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

font_path = "Arial.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 32)


model_path = "sign_languge_model.keras"  
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Đã tải mô hình thành công!")
else:
    print("Không tìm thấy mô hình! Kiểm tra lại file 'sign_languge_model.keras'.")
    exit()


label_encoder = sorted(os.listdir("sign_language_data"))

reverse_replacements = {
    "slash": "/",
    "backslash": "\\",
    "question": "?",
    "tilde": "~",
    "dot": "."
}

accent_map = {
    ("A", "^"): "Â",
    ("E", "^"): "Ê",
    ("O", "^"): "Ô",
    ("U", "'"): "Ư",
    ("O", "'"): "Ơ",

    ("A", "w"): "Ă",

    ("A", "."): "Ạ",
    ("A", "/"): "Á",
    ("A", "\\"): "À",
    ("A", "~"): "Ã",
    ("A", "?"): "Ả",

    ("E", "."): "Ẹ",
    ("E", "/"): "É",
    ("E", "\\"): "È",
    ("E", "~"): "Ẽ",
    ("E", "?"): "Ẻ",

    ("O", "."): "Ọ",
    ("O", "/"): "Ó",
    ("O", "\\"): "Ò",
    ("O", "~"): "Õ",
    ("O", "?"): "Ỏ",

    ("U", "."): "Ụ",
    ("U", "/"): "Ú",
    ("U", "\\"): "Ù",
    ("U", "~"): "Ũ",
    ("U", "?"): "Ủ",

    ("I", "."): "Ị",
    ("I", "/"): "Í",
    ("I", "\\"): "Ì",
    ("I", "~"): "Ĩ",
    ("I", "?"): "Ỉ",

    ("Y", "."): "Ỵ",
    ("Y", "/"): "Ý",
    ("Y", "\\"): "Ỳ",
    ("Y", "~"): "Ỹ",
    ("Y", "?"): "Ỷ",

    ("Â", "."): "Ậ",
    ("Â", "/"): "Ấ",
    ("Â", "\\"): "Ầ",
    ("Â", "~"): "Ẫ",
    ("Â", "?"): "Ẩ",

    ("Ê", "."): "Ệ",
    ("Ê", "/"): "Ế",
    ("Ê", "\\"): "Ề",
    ("Ê", "~"): "Ễ",
    ("Ê", "?"): "Ể",

    ("Ô", "."): "Ộ",
    ("Ô", "/"): "Ố",
    ("Ô", "\\"): "Ồ",
    ("Ô", "~"): "Ỗ",
    ("Ô", "?"): "Ổ",

    ("Ơ", "."): "Ợ",
    ("Ơ", "/"): "Ớ",
    ("Ơ", "\\"): "Ờ",
    ("Ơ", "~"): "Ỡ",
    ("Ơ", "?"): "Ở",

    ("Ư", "."): "Ự",
    ("Ư", "/"): "Ứ",
    ("Ư", "\\"): "Ừ",
    ("Ư", "~"): "Ữ",
    ("Ư", "?"): "Ử",

    ("Ă", "."): "Ặ",
    ("Ă", "/"): "Ắ",
    ("Ă", "\\"): "Ằ",
    ("Ă", "~"): "Ẵ",
    ("Ă", "?"): "Ẳ"
}


def combine_vietnamese_characters():
    global result_text

    if not result_text or (result_text[-1] in {".", "/", "\\", "~", "?"} and len(result_text) == 1):
        result_text = result_text[:-1]  
        return

    if len(result_text) >= 2 and result_text[-1] in {".", "/", "\\", "~", "?"}:
        if result_text[-2] not in {key[0] for key in accent_map}:  
            result_text = result_text[:-1]  
            return

    if len(result_text) >= 2:
        last_two = (result_text[-2], result_text[-1])
        if last_two in accent_map:
            result_text = result_text[:-2] + accent_map[last_two]
            return
    if len(result_text) >= 3:
        last_three = (result_text[-3], result_text[-2], result_text[-1])
        if last_three in accent_map:
            result_text = result_text[:-3] + accent_map[last_three]
            return   
def decode_label(predicted_label):
    """Chuyển nhãn dự đoán về ký hiệu gốc"""
    return reverse_replacements.get(predicted_label, predicted_label)

import unicodedata

def remove_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
                   if unicodedata.category(c) != 'Mn')


import tkinter as tk
def delete_last_character():
    global result_text, delete_running, delete_active
    
    if result_text:  
        print(f"Deleting character. Current text: {result_text}")
        result_text = result_text[:-1]
        update_text_display()

    else:
        print("No more characters to delete.")
        delete_running = False  
        delete_active = True 


def update_result_box(new_text, append=True):
    global result_text, delete_running

    if not append:
        result_text = ""
        text_box.configure(state="normal")
        text_box.delete("1.0", tk.END)
        text_box.configure(state="disabled")
        delete_running = False  
        return

    prev_text = result_text  

    if new_text == "space":
        result_text += " "
        delete_running = False  
    elif new_text == "del":
        if not delete_running:  
            delete_running = True
            root.after(1000, delete_last_character)  
    else:
        result_text += new_text
        delete_running = False 

    if result_text != prev_text:
        combine_vietnamese_characters()
        update_text_display()
        
cursor_visible = True
def update_text_display():
    """ Cập nhật hiển thị văn bản với dấu nháy chính xác """
    global cursor_visible
    
    text_box.configure(state="normal")
    text_box.delete("1.0", tk.END)

    cursor_symbol = "\u23B8"  
    display_text = result_text if result_text else ""
    display_text = display_text.lower()
    
    text_box.insert(tk.END, display_text + cursor_symbol if cursor_visible else display_text)
    
    text_box.see(tk.END)
    text_box.configure(state="disabled")

    cursor_visible = not cursor_visible
    root.after(1000, update_text_display)  

def draw_text(img, text, position, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def normalize_landmarks(landmarks):
    wrist = np.array(landmarks[0])
    palm_center = np.mean([landmarks[5],landmarks[9], landmarks[13], landmarks[17]], axis=0)
    base_distance = np.linalg.norm(wrist - palm_center)
    return (landmarks - wrist) / base_distance if base_distance > 0 else np.zeros_like(landmarks)

result_text = ""
delete_running = False  
delete_active = False
last_appended_prediction = None
sequence = []
frame_threshold = 30
running = False

def process_video_with_prediction():
    global cap, running, last_appended_prediction, sequence, delete_running, delete_active
    if cap is not None and running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(frame_rgb)
            prediction_text = " "
            landmark_img = np.ones((480, 640, 3), dtype=np.uint8) * 255

            delete_active = False
            sequence_length =30
            frame_hands_data = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(landmark_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    hand_data = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    norm_hand_data = normalize_landmarks(hand_data)
                    frame_hands_data.append(norm_hand_data.flatten())

                if len(frame_hands_data) == 1:
                    frame_hands_data.append(np.zeros(63))
                elif len(frame_hands_data) == 0:
                    frame_hands_data = [np.zeros(63), np.zeros(63)]
                
                frame_hands_data = np.concatenate(frame_hands_data)

                if frame_hands_data.shape == (126,):
                    sequence.append(frame_hands_data)
                    
                if len(sequence) == frame_threshold:
                    input_data = np.array(sequence)
                    if input_data.shape == (frame_threshold, 126):
                        input_data = np.expand_dims(input_data, axis=0)
                        input_data = (input_data - 0.5) * 2
                        prediction = model.predict(input_data, verbose=0)
                        predicted_label = label_encoder[np.argmax(prediction)]
                        confidence = np.max(prediction)
                        
                        if confidence >= 0.9:
                            decoded_label = decode_label(predicted_label)
                            prediction_text = f"{decoded_label} ({confidence * 100:.2f}%)"

                            if decoded_label == "del":
                                    delete_active = True  
                                    if not delete_running:
                                        delete_running = True
                                        root.after(1000, delete_last_character) 
                            else:
                                delete_running = False 

                            if decoded_label != last_appended_prediction:
                                update_result_box(decoded_label, append=True)
                                last_appended_prediction = decoded_label
                    sequence.clear()   
                else:
                    delete_running = False  

                predict_label.config(text=prediction_text)
            
            frame_pil = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(frame_pil)
            draw.text((20, 50), prediction_text, font=font, fill=(0, 255, 0))
            frame_rgb = np.array(frame_pil)
            
            predict_label.config(text=prediction_text)
            
            img_landmark = Image.fromarray(landmark_img)
            img_landmark_tk = ImageTk.PhotoImage(image=img_landmark)
            label_landmark.imgtk = img_landmark_tk
            label_landmark.config(image=img_landmark_tk)
            
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)
        
        root.after(10, process_video_with_prediction)

def speak_text():
    text = text_box.get("1.0", tk.END).strip()
    if text:
        tts = gTTS(text=text, lang='vi')
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0) 

        pygame.mixer.quit()
        pygame.mixer.init()

        def play_audio():
            pygame.mixer.music.load(audio_stream, "mp3")
            pygame.mixer.music.play()

        threading.Thread(target=play_audio, daemon=True).start()

def stop_video():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
        cap = None

box1 = LabelFrame(root, text="Video / Camera", padx=2, pady=2)
box1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

label = tk.Label(box1)
label.pack(fill="both", expand=True)

box2 = LabelFrame(root, text="Landmark & Prediction", padx=5, pady=5)
box2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
label_landmark = tk.Label(box2)
label_landmark.pack(fill="both", expand=True)
predict_label = tk.Label(box2, text="Waiting for Prediction...", font=("Arial", 12))
predict_label.pack(fill="both", expand=True)

box3 = LabelFrame(root, text="Result", padx=5, pady=5)
box3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
text_box = tk.Text(box3, height=10, width=40)
text_box.pack(fill="both", expand=True)
btn_speak = tk.Button(box3, text="🔊 Đọc văn bản", width=15, height=2, command=speak_text)
btn_speak.pack(pady=5)

box4 = LabelFrame(root, text="Action", padx=10, pady=10)
box4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

box4.columnconfigure((0, 1), weight=1)

btn_camera = tk.Button(box4, text="📷 Open Camera", width=15, height=2, command=open_camera)
btn_camera.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

btn_video = tk.Button(box4, text="🎥 Choose Video", width=15, height=2, command=open_video)
btn_video.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

btn_clear = tk.Button(box4, text="❌ Delete all text", width=15, height=2, command=clear_text)
btn_clear.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

btn_save = tk.Button(box4, text="💾 Save text", width=15, height=2, command=save_text)
btn_save.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
btn_open_app1_BCC = tk.Button(box4, text="Alphabet and hand gesture guide", width=15, height=2,command=open_app1_BCC)
btn_open_app1_BCC.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
root.mainloop()