import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    global img, processed_img
    img = cv2.imread(file_path)
    processed_img = img.copy()
    display_image(img)

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    max_width = 500 
    max_height = 500 
    image.thumbnail((max_width, max_height))

    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image

def preprocess_image(method):
    global img, processed_img

    if img is None:
        messagebox.showwarning("Input Error", "Please load an image first.")
        return

    if method == "grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    elif method == "histogram":
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        processed_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif method == "edge_detection":
        edges = cv2.Canny(img, 100, 200)
        processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif method == "blur":
        processed_img = cv2.GaussianBlur(img, (5, 5), 0)

    display_image(processed_img)

def detect_motor():
    if processed_img is None:
        messagebox.showwarning("Input Error", "Please load and preprocess an image first.")
        return

    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"

    if not os.path.exists(cfg_path):
        messagebox.showerror("File Error", f"File {cfg_path} not found.")
        return

    if not os.path.exists(weights_path):
        messagebox.showerror("File Error", f"File {weights_path} not found.")
        return

    yolo_net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    height, width, _ = processed_img.shape
    blob = cv2.dnn.blobFromImage(processed_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    motor_boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "motorbike":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                motor_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(motor_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in range(len(motor_boxes)):
        if i in indexes:
            x, y, w, h = motor_boxes[i]
            motor_type = classes[class_ids[i]]
            
            thickness = 4
            cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            cv2.putText(processed_img, motor_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    display_image(processed_img)

    if len(motor_boxes) > 0:
        result_label.config(text="Motor Detected!")
    else:
        result_label.config(text="No Motor Detected")

root = tk.Tk()
root.title("Motor Detection App")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

side_frame = tk.Frame(main_frame)
side_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

button_width = 20

load_button = tk.Button(side_frame, text="Load Image", font=("Arial", 12), width=button_width, command=load_image)
load_button.pack(pady=5)

grayscale_button = tk.Button(side_frame, text="Grayscale", font=("Arial", 12), width=button_width, command=lambda: preprocess_image("grayscale"))
grayscale_button.pack(pady=5)

histogram_button = tk.Button(side_frame, text="Histogram Equalization", font=("Arial", 12), width=button_width, command=lambda: preprocess_image("histogram"))
histogram_button.pack(pady=5)

edge_button = tk.Button(side_frame, text="Edge Detection", font=("Arial", 12), width=button_width, command=lambda: preprocess_image("edge_detection"))
edge_button.pack(pady=5)

blur_button = tk.Button(side_frame, text="Gaussian Blur", font=("Arial", 12), width=button_width, command=lambda: preprocess_image("blur"))
blur_button.pack(pady=5)

detect_button = tk.Button(side_frame, text="Detect Motor", font=("Arial", 12), width=button_width, command=detect_motor)
detect_button.pack(pady=5)

graphics_frame = tk.Frame(main_frame)
graphics_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
image_label = tk.Label(graphics_frame)
image_label.pack()

result_label = tk.Label(main_frame, text="", font=("Arial", 12), fg="blue")
result_label.pack(side=tk.BOTTOM, pady=10)

img = None
processed_img = None

root.mainloop()

