import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os

# ------------------ Suppress TensorFlow logs ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ------------------ Import prediction functions ------------------
from predict import predict_from_webcam, predict_from_image_file

# ------------------ GUI SETUP ------------------
root = tk.Tk()
root.title("Handwritten Digit Recognition using CNN")
root.geometry("480x260")
root.resizable(False, False)

# ------------------ CENTER FRAME ------------------
main = tk.Frame(root)
main.place(relx=0.5, rely=0.5, anchor="center")

# ------------------ THREAD WRAPPER ------------------
def run_async(func, *args):
    threading.Thread(target=lambda: func(*args), daemon=True).start()

# ------------------ ACTIONS ------------------
def run_webcam():
    run_async(predict_from_webcam)

def run_image():
    path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if path:
        run_async(predict_from_image_file, path)

# ------------------ UI ELEMENTS ------------------

# Title
tk.Label(main, text="Handwritten Digit Recognition using CNN",
         font=("Segoe UI", 12, "bold")).pack(pady=(0, 12))

# Webcam section
ttk.Button(main, text="Use Live Webcam", command=run_webcam, width=25).pack()
tk.Label(main,
         text="Capture digits live using webcam",
         fg="gray").pack(pady=(2, 10))

# Separator
ttk.Separator(main, orient="horizontal").pack(fill="x", pady=5)

# Image section
ttk.Button(main, text="Select Image File", command=run_image, width=25).pack(pady=(10, 0))
tk.Label(main,
         text="Select an image containing handwritten digits",
         fg="gray").pack(pady=(2, 10))

# Footer
tk.Label(root,
         text="Developed by ShivCK",
         fg="gray").pack(side="bottom", pady=6)

# ------------------ RUN ------------------
root.mainloop()