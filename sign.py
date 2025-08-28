import tkinter as tk
from tkinter import messagebox
import cv2
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

#Load your trained model (change path accordingly)
model = load_model("sign_language_model.keras")

#Example class names (update as per your model)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del','nothing']

root = tk.Tk()
root.title("Sign Language Detection")
root.configure(bg='blue')

cap = None
running = False

label = tk.Label(root)
label.pack(padx=10, pady=10)

prediction_label = tk.Label(root, text="", font=("Helvetica", 20), bg='blue', fg='white')
prediction_label.pack(pady=10)

def preprocess_frame(frame):
    # Resize and normalize the frame as your model expects
    img = cv2.resize(frame, (64, 64))  # Change size to your model's input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def within_time_window():
    now = datetime.now().time()
    start_time = datetime.strptime("18:00", "%H:%M").time()  # 6 PM
    end_time = datetime.strptime("22:00", "%H:%M").time()    # 10 PM
    return start_time <= now <= end_time


def show_frame():
    global cap, running
    if running and cap is not None:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to grab frame from camera.")
            stop_camera()
            return

        # Flip frame horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Prediction
        img = preprocess_frame(frame)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        prediction_label.config(text=f"Prediction: {predicted_class} ({confidence*100:.2f}%)")

        # Show camera frame in GUI
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label.imgtk = img_tk
        label.configure(image=img_tk)

        root.after(10, show_frame)  # call show_frame again after 10ms

def start_camera():
    global cap, running
    if not within_time_window():
        messagebox.showwarning("Time Restriction", "App only works between 6 PM and 10 PM.")
        return
    cap = cv2.VideoCapture(0)
    running = True
    show_frame()


def stop_camera():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
        cap = None
    label.config(image='')  # Clear image
    prediction_label.config(text="")

btn_frame = tk.Frame(root, bg='blue')
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="Open Camera", command=start_camera, width=15)
start_btn.grid(row=0, column=0, padx=5)

stop_btn = tk.Button(btn_frame, text="Stop Camera", command=stop_camera, width=15)
stop_btn.grid(row=0, column=1, padx=5)

root.mainloop()
