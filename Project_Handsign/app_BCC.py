import cv2
import os
import tkinter as tk
from tkinter import Canvas, Frame, Scrollbar, Listbox, Button, Toplevel, Label, RIGHT, Y, BOTH, END
from PIL import Image, ImageTk

r = tk.Tk()
r.geometry("900x500")
r.title("Media Player")
r.configure(bg="white")

VIDEO_FOLDER = "Bang_chu_cai"

class MediaPlayer:
    def __init__(self, canvas, pause_button):
        self.canvas = canvas
        self.current_player = None
        self.pause_button = pause_button

    def play_video(self, video_path):
        if self.current_player:
            self.current_player.playing = False  

        self.current_player = VideoPlayer(self.canvas, video_path, self.pause_button)
        self.current_player.play_video()

    def pause_video(self):
        if self.current_player:
            self.current_player.pause_video()

    def slow_motion(self):
        if self.current_player:
            self.current_player.update_delay = 100 

    def normal_speed(self):
        if self.current_player:
            self.current_player.update_delay = 30  

class VideoPlayer:
    def __init__(self, canvas, video_path, pause_button):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.canvas = canvas
        self.playing = False
        self.update_delay = 30  
        self.pause_button = pause_button

    def play_video(self):
        if not self.video.isOpened():
            print(f"Không thể mở video: {self.video_path}")
            return
        
        if not self.playing:
            self.playing = True
            self.pause_button.config(text="Pause")
            self.update_video()

    def pause_video(self):
        self.playing = not self.playing
        self.pause_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.update_video()

    def update_video(self):
        if self.playing:
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (500, 400))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk
                self.canvas.after(self.update_delay, self.update_video)
            else:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)  

def show_image_window(image_path):
    img_window = Toplevel(r)
    img_window.title("Xem ảnh")
    img = Image.open(image_path)
    img = img.resize((800, 700))
    imgtk = ImageTk.PhotoImage(img)
    lbl_img = Label(img_window, image=imgtk)
    lbl_img.image = imgtk
    lbl_img.pack()

all_files = [os.path.join(VIDEO_FOLDER, f) for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4", ".png"))]

def on_item_select(event):
    selected_index = listbox.curselection()
    if selected_index:
        item_index = selected_index[0]
        item_path = all_files[item_index]  

        if item_path.endswith(".mp4"):
            media_player.play_video(item_path)  
            btn_view_image.pack_forget()  
        elif item_path.endswith(".png"):
            btn_view_image.config(command=lambda: show_image_window(item_path))
            btn_view_image.pack()  

left_frame = Frame(r, width=200, height=500, bg="white")
left_frame.pack(side="left", fill="both")

scrollbar = Scrollbar(left_frame)
scrollbar.pack(side=RIGHT, fill=Y)

listbox = Listbox(left_frame, yscrollcommand=scrollbar.set, width=30, height=20)
listbox.pack(fill=BOTH, expand=True)
scrollbar.config(command=listbox.yview)

right_frame = Frame(r, width=600, height=500, bg="white")
right_frame.pack(side="right", fill="both", expand=True)

video_canvas = Canvas(right_frame, width=500, height=400, bg="black")
video_canvas.pack(pady=20)

btn_frame = Frame(right_frame, bg="white")
btn_frame.pack()

btn_pause = Button(btn_frame, text="Pause", bg="lightgray")
btn_pause.grid(row=0, column=1, padx=5)

btn_slow = Button(btn_frame, text="Slow Motion", bg="lightgray")
btn_slow.grid(row=0, column=2, padx=5)

btn_normal = Button(btn_frame, text="Normal", bg="lightgray")
btn_normal.grid(row=0, column=3, padx=5)

btn_view_image = Button(right_frame, text="Xem ảnh", bg="lightgray")
btn_view_image.pack()
btn_view_image.pack_forget()

media_player = MediaPlayer(video_canvas, btn_pause)

btn_pause.config(command=media_player.pause_video)
btn_slow.config(command=media_player.slow_motion)
btn_normal.config(command=media_player.normal_speed)

listbox.delete(0, END)
for path in all_files:
    listbox.insert(END, os.path.basename(path))

listbox.bind("<<ListboxSelect>>", on_item_select)

r.mainloop()
