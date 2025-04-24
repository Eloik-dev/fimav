import tkinter as tk
from PIL import Image, ImageTk
import cv2


class MainWindow(tk.Tk):
    def __init__(self, video_capture, detector, face_size, width, height):
        super().__init__()
        self.title("Video Feed")
        self.video_capture = video_capture
        self.width = width
        self.height = height
        self.detector = detector
        self.face_size = face_size

        self.video_frame = tk.Label(self)
        self.video_frame.pack()

        self.photo = None
        self.interval = round((1 / 30) * 1000)

        self.update_frame()

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()
        if frame is None:
            self.after(self.interval, self.update_frame)
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
        img = Image.fromarray(frame)
        img = img.resize((1920, 1080), Image.LANCZOS)  # Resize for display
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_frame.config(image=img_tk)
        self.video_frame.image = img_tk  # Keep a reference!

        self.after(self.interval, self.update_frame)

