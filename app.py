
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import datetime
import csv
import os
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model(r"E:/Taiwan Internship/Socket project/NEW SOCKET APP/hole_detector_final_model123.h5")

CLASSES = ["Upside", "Downside"]
LOG_FILE = "orientation_log.csv"
os.makedirs("captured_downside", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Orientation"])

def preprocess(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

def predict_orientation(frame):
    processed = preprocess(frame)
    preds = model.predict(processed, verbose=0)[0]
    return CLASSES[np.argmax(preds)]

def save_downside_frame(frame, orientation):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_downside/{orientation}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def log_downside(orientation):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, orientation])

class SocketApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Socket Orientation Detector")
        self.root.geometry("1280x720")
        self.root.configure(bg="white")

        self.video = cv2.VideoCapture(0)
        self.last_logged_time = 0
        self.total_count = 0
        self.proper_count = 0
        self.improper_count = 0

        # Layout configuration
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left Frame - Live Feed
        self.left_frame = Frame(self.root, bg="black", bd=2, relief="sunken")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.live_label = Label(self.left_frame)
        self.live_label.pack(expand=True, fill="both")

        # Right Frame - Socket Report
        self.right_frame = Frame(self.root, bg="white")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        Label(self.right_frame, text="Socket Report", font=("Arial", 16, "bold"), bg="white").pack(pady=(10, 5), anchor="w")
        self.status = Label(self.right_frame, text="Status: Analyzing...", fg="blue", bg="white")
        self.status.pack(anchor="w")
        self.conveyor = Label(self.right_frame, text="Conveyor: Running", fg="blue", bg="white")
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

        # Buttons
        Button(self.right_frame, text="↑ Start Conveyor", command=self.start_conveyor, width=25).pack(pady=5)
        Button(self.right_frame, text="↓ Stop Conveyor", command=self.stop_conveyor, width=25).pack(pady=5)
        Button(self.right_frame, text="\U0001F4BE Export Excel", command=self.export_excel, width=25).pack(pady=5)
        Button(self.right_frame, text="⚙ PLC Settings", command=self.plc_settings, width=25).pack(pady=5)

        self.update_frame()

    def start_conveyor(self):
        self.conveyor.config(text="Conveyor: Running", fg="blue")

    def stop_conveyor(self):
        self.conveyor.config(text="Conveyor: Stopped", fg="red")

    def export_excel(self):
        os.system(f'start excel.exe "{LOG_FILE}"')  # Windows only

    def plc_settings(self):
        print("Open PLC Settings")  # Placeholder for GPIO or serial setup

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            orientation = predict_orientation(frame)
            self.orientation.config(text=f"Orientation: {orientation}")

            self.total_count += 1
            if orientation == "Upside":
                self.proper_count += 1
            else:
                self.improper_count += 1
                current_time = datetime.datetime.now().timestamp()
                if current_time - self.last_logged_time > 2:
                    log_downside(orientation)
                    save_downside_frame(frame, orientation)
                    self.last_logged_time = current_time

            # Update counts
            self.total.config(text=f"Total Sockets Detected: {self.total_count}")
            self.proper.config(text=f"Properly Oriented Count: {self.proper_count}")
            self.improper.config(text=f"Improperly Oriented Count: {self.improper_count}")

            # Draw border
            border_color = (0, 255, 0) if orientation == "Upside" else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 10)

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
