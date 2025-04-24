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

        self.label = tk.Label(self, highlightthickness=0)
        self.label.pack()

        self.photo = None
        self.interval = round((1 / 30) * 1000)

        self.update_frame()

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()
        if frame is None:
            self.after(self.interval, self.update_frame)
            return

        frame = cv2.resize(frame, (self.width, self.height))
        data = frame[..., ::-1].tobytes()
        pil_img = Image.frombuffer(
            "RGB", (self.width, self.height), data, "raw", "RGB", 0, 1
        )
        self.photo = ImageTk.PhotoImage(pil_img, master=self.label)
        self.label.config(image=self.photo)

        self.after(self.interval, self.update_frame)

