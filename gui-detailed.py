import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
import cv2
import time

# ------------------ Suppress TensorFlow logs ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ------------------ Import prediction functions ------------------
from predict import (
    predict_from_webcam,
    predict_from_image_file,
    WINDOW_IMAGE,
    WINDOW_WEBCAM
)

# ------------------ GUI SETUP ------------------
root = tk.Tk()
root.title("Handwritten Digit Recognition using CNN")
root.geometry("720x640")
root.resizable(False, False)

# ------------------ Styles ------------------
style = ttk.Style(root)
style.theme_use("vista")

style.configure(
    "Green.Horizontal.TProgressbar",
    troughcolor="#e8f5e9",
    background="#2e7d32",
    thickness=18
)

style.configure(
    "Red.Horizontal.TProgressbar",
    troughcolor="#fdecea",
    background="#c62828",
    thickness=18
)

style.configure(
    "Loader.Horizontal.TProgressbar",
    troughcolor="#f0f0f0",
    background="#4a6fa5",
    thickness=6
)

# ------------------ UI ------------------
container = ttk.Frame(root, padding=20)
container.pack(expand=True, fill="both")

title = ttk.Label(
    container,
    text="Handwritten Digit Recognition using CNN",
    font=("Segoe UI", 15, "bold")
)
title.pack(pady=(0, 12))

# ================== MAIN GRID ==================
main_frame = ttk.Frame(container)
main_frame.pack(expand=True, fill="both")

main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=2)

# ================== LEFT PANEL ==================
left_panel = ttk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="n", padx=(0, 20))

# ---- Instruction text JUST ABOVE BUTTONS ----
ttk.Label(
    left_panel,
    text="Choose an option below to start digit recognition:",
    font=("Segoe UI", 9),
    wraplength=220,
    justify="left"
).pack(pady=(0, 12))

# ---- Buttons ----
ttk.Button(left_panel, text="Open Webcam", width=22, command=lambda: run_webcam()).pack(pady=(0, 5))
ttk.Label(
    left_panel,
    text="Live digit recognition from camera",
    font=("Segoe UI", 9),
    foreground="gray"
).pack(pady=(0, 15))

ttk.Button(left_panel, text="Choose Image File", width=22, command=lambda: run_image()).pack(pady=(0, 5))
ttk.Label(
    left_panel,
    text="Predict digit from a selected image",
    font=("Segoe UI", 9),
    foreground="gray"
).pack(pady=(0, 15))

# ------------------ Compact Loader ------------------
loader_frame = ttk.Frame(left_panel)

status_label = ttk.Label(
    loader_frame,
    text="",
    font=("Segoe UI", 9, "italic"),
    foreground="gray"
)
status_label.pack(pady=(10, 4))

progress = ttk.Progressbar(
    loader_frame,
    mode="indeterminate",
    length=120,
    style="Loader.Horizontal.TProgressbar"
)
progress.pack()

def show_loader(text):
    status_label.config(text=text)
    progress.start(10)
    loader_frame.pack()
    root.update_idletasks()

def hide_loader():
    progress.stop()
    loader_frame.pack_forget()
    status_label.config(text="")

# ================== RIGHT PANEL ==================
right_panel = ttk.Frame(main_frame)
right_panel.grid(row=0, column=1, sticky="nsew")
right_panel.columnconfigure(0, weight=1)

# -------- Helper for Metric Row --------
def metric_row(parent, label, value, style, max_value=1.0):
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=6)

    ttk.Label(row, text=label, width=16).pack(side="left")

    bar = ttk.Progressbar(
        row,
        value=(value / max_value) * 100,
        maximum=100,
        style=style
    )
    bar.pack(side="left", fill="x", expand=True, padx=6)

    ttk.Label(row, text=f"{value*100:.2f}%", width=8, anchor="e").pack(side="right")

# ------------------ Train Metrics Card ------------------
train_card = ttk.Frame(right_panel, padding=10)
train_card.pack(fill="x", pady=(0, 10))

ttk.Label(train_card, text="Training Metrics", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
metric_row(train_card, "Accuracy", 0.9836, "Green.Horizontal.TProgressbar")
metric_row(train_card, "Loss", 0.0633, "Red.Horizontal.TProgressbar")

# ------------------ Validation Metrics Card ------------------
val_card = ttk.Frame(right_panel, padding=10)
val_card.pack(fill="x", pady=(0, 10))

ttk.Label(val_card, text="Validation Metrics", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
metric_row(val_card, "Accuracy", 0.9946, "Green.Horizontal.TProgressbar")
metric_row(val_card, "Loss", 0.0198, "Red.Horizontal.TProgressbar")

# ------------------ Model Info Card ------------------
info_card = ttk.Frame(right_panel, padding=10)
info_card.pack(fill="x", pady=(0, 10))

ttk.Label(info_card, text="Model Information", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

info_text = (
    "Batch Size     : 64\n"
    "Epochs         : 10\n\n"
    "Total Params   : 413,514 (1.58 MB)\n"
    "Trainable      : 413,514\n"
    "Non-trainable  : 0"
)
ttk.Label(info_card, text=info_text, font=("Segoe UI", 9), justify="left").pack(anchor="w")

# ------------------ OpenCV Window Watcher ------------------
def watch_cv_window(window_name):
    def poll():
        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    root.after(0, hide_loader)
                    break
            except cv2.error:
                pass
            time.sleep(0.2)
    threading.Thread(target=poll, daemon=True).start()

# ------------------ Run Predictions ------------------
def run_webcam():
    show_loader("Opening webcam…")
    def task():
        watch_cv_window(WINDOW_WEBCAM)
        predict_from_webcam()
    threading.Thread(target=task, daemon=True).start()

def run_image():
    path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not path:
        return
    show_loader("Processing image…")
    def task():
        watch_cv_window(WINDOW_IMAGE)
        predict_from_image_file(path)
    threading.Thread(target=task, daemon=True).start()

# ------------------ Footer ------------------
footer = ttk.Label(
    root,
    text="Developed by ShivCK with TensorFlow, OpenCV, Tkinter using Python",
    font=("Segoe UI", 8)
)
footer.pack(side="bottom", pady=6)

# ------------------ START ------------------
root.mainloop()