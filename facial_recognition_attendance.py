import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
attendance_records = []
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
root = tk.Tk()
root.title("Facial Recognition Attendance System")
def register_face():
    name = simpledialog.askstring("Register Face", "Enter Name:")
    if not name:
        return
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))
            cv2.imwrite(os.path.join(dataset_path, f"{name}.jpg"), face_crop)
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Success", f"Face registered for {name}")
            return
        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
def recognize_faces():
    known_faces = {}
    for file_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, file_name)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (200, 200))
            name = os.path.splitext(file_name)[0]
            known_faces[name] = img_resized
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))
            name = "Unknown"
            for registered_name, registered_face in known_faces.items():
                if np.mean(np.abs(face_crop - registered_face)) < 50:
                    name = registered_name
                    save_attendance(name)
                    break
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Face Recognition Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def save_attendance(name):
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    attendance_records.append((name, formatted_time))
    print(f"{name} marked present at {formatted_time}")
def view_attendance():
    records = "\n".join([f"{name} - {time}" for name, time in attendance_records])
    messagebox.showinfo("Attendance Records", records or "No records found.")
tk.Button(root, text="Register Face", command=register_face).pack(pady=10)
tk.Button(root, text="Recognize Faces", command=recognize_faces).pack(pady=10)
tk.Button(root, text="View Attendance", command=view_attendance).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit).pack(pady=10)
root.mainloop()