import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import datetime
import csv
import os
import threading
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# Constants
CLASSES = ["Hole", "Non-hole"]
IMG_SIZE = (224, 224, 3)
MIN_CONFIDENCE = 0.997
TFLITE_MODEL_PATH = "hole_detector_final_model123.tflite"
LOG_FILE = "orientation_log.csv"
os.makedirs("captured_downside", exist_ok=True)

# Create log file if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Orientation"])


# Load TFLite model
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# Preprocess frame for TFLite
def preprocess(frame):
    resized = cv2.resize(frame, IMG_SIZE[:2])
    normalized = resized / 255.0
    return np.expand_dims(normalized.astype(np.float32), axis=0)


class SocketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Socket Orientation Detector")
        self.root.geometry("1280x720")
        self.root.configure(bg="white")

        self.model = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        self.video = cv2.VideoCapture(0)
        self.last_logged_time = 0
        self.total_count = 0
        self.proper_count = 0
        self.improper_count = 0

        self.setup_ui()
        threading.Thread(target=self.load_model_thread, daemon=True).start()
        self.update_frame()

    def setup_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.left_frame = Frame(self.root, bg="black", bd=2, relief="sunken")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.live_label = Label(self.left_frame)
        self.live_label.pack(expand=True, fill="both")

        self.right_frame = Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        Label(self.right_frame, text="Socket Report", font=("Arial", 16, "bold"), bg="white").pack(pady=(10, 5), anchor="w")
        self.status = Label(self.right_frame, text="Status: Loading model...", fg="orange", bg="white")
        self.status.pack(anchor="w")
        self.conveyor = Label(self.right_frame, text="Conveyor: Stopped", fg="red", bg="white")
        self.conveyor.pack(anchor="w")
        self.orientation = Label(self.right_frame, text="Orientation: -", fg="blue", bg="white")
        self.orientation.pack(anchor="w")
        self.size = Label(self.right_frame, text="Size: Medium", fg="blue", bg="white")
        self.size.pack(anchor="w")
        self.total = Label(self.right_frame, text="Total Sockets Detected: 0", bg="white")
        self.total.pack(anchor="w")
        self.proper = Label(self.right_frame, text="Properly Oriented Count: 0", bg="white")
        self.proper.pack(anchor="w")
        self.improper = Label(self.right_frame, text="Improperly Oriented Count: 0", bg="white")
        self.improper.pack(anchor="w")

        Button(self.right_frame, text="↑ Start Conveyor", command=self.start_conveyor, width=25).pack(pady=5)
        Button(self.right_frame, text="↓ Stop Conveyor", command=self.stop_conveyor, width=25).pack(pady=5)
        Button(self.right_frame, text="\U0001F4BE Export Excel", command=self.export_excel, width=25).pack(pady=5)
        Button(self.right_frame, text="⚙ PLC Settings", command=self.plc_settings, width=25).pack(pady=5)

    def load_model_thread(self):
        try:
            start = time.time()
            self.model, self.input_details, self.output_details = load_tflite_model()
            end = time.time()
            self.model_loaded = True
            self.status.config(text=f"Status: Model Loaded ({end - start:.2f}s)", fg="green")
        except Exception as e:
            self.status.config(text="Status: Model Load Failed", fg="red")
            print("[ERROR] Model load failed:", e)

    def start_conveyor(self):
        self.conveyor.config(text="Conveyor: Running", fg="blue")

    def stop_conveyor(self):
        self.conveyor.config(text="Conveyor: Stopped", fg="red")

    def export_excel(self):
        os.system(f'start excel.exe "{LOG_FILE}"')  # On Pi, this may need to be handled differently

    def plc_settings(self):
        print("Open PLC Settings")

    def predict_orientation(self, frame):
        processed = preprocess(frame)
        self.model.set_tensor(self.input_details[0]['index'], processed)
        self.model.invoke()
        pred = self.model.get_tensor(self.output_details[0]['index'])[0][0]

        confidence = pred if pred > 0.5 else 1 - pred
        if confidence >= MIN_CONFIDENCE:
            label_idx = int(pred > 0.5)
            label = CLASSES[label_idx]
            return f"{label} ({confidence * 100:.2f}%)", label, confidence
        else:
            return "Uncertain (<99.65%)", "Uncertain", confidence

    def log_downside(self, orientation):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, orientation])

    def save_downside_frame(self, frame, orientation):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_downside/{orientation}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    def update_frame(self):
        ret, frame = self.video.read()
        if ret and self.model_loaded:
            result_str, label, confidence = self.predict_orientation(frame)
            current_time = datetime.datetime.now().timestamp()

            if label == "Hole":
                self.total_count += 1
                self.proper_count += 1
                if current_time - self.last_logged_time > 2:
                    self.log_downside(label)
                    self.save_downside_frame(frame, label)
                    self.last_logged_time = current_time
                color = (0, 255, 0)
            elif label == "Non-hole":
                self.improper_count += 1
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            self.orientation.config(text=f"Orientation: {result_str}")
            self.total.config(text=f"Total Sockets Detected: {self.total_count}")
            self.proper.config(text=f"Properly Oriented Count: {self.proper_count}")
            self.improper.config(text=f"Improperly Oriented Count: {self.improper_count}")

            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.live_label.imgtk = imgtk
            self.live_label.configure(image=imgtk)

        self.root.after(100, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = SocketApp(root)
    root.mainloop()
