import tkinter as tk
from tkinter import Canvas, Label
from PIL import Image, ImageTk
import cv2


class MainWindow(tk.Tk):
    def __init__(self, video_capture, detector, width, height):
        super().__init__()
        self.title("Face & Emotion Detection")
        self.video_capture = video_capture
        self.detector = detector

        # Setup canvas for video
        self.width = width
        self.height = height
        self.canvas = Canvas(self, width=width, height=height)
        self.canvas.pack()

        # Label for emotion text
        self.emotion_label = Label(
            self.canvas, text="", font=(None, 24, "bold"), bg="black", fg="white"
        )
        self.canvas.create_window(
            self.width // 2, 30, window=self.emotion_label, anchor="n"
        )

        # PhotoImage reference
        self.photo = None

        # Start update loop
        self.after(33, self.update_frame)

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()

        if frame is None:
            self.after(33, self.update_frame)
            return

        current_detections = self.detector.get_latest_detection()

        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw rectangles
        for x, y, w, h in current_detections:
            cv2.rectangle(
                rgb,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        # Convert to Tk image
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Update label
        current_emotion = self.detector.get_current_emotion()
        if current_emotion:
            self.emotion_label.config(
                text=f"Vous êtes: {current_emotion}",
                fg="green",
                font=(None, 24, "bold", "italic"),
            )
        else:
            self.emotion_label.config(
                text="Aucune émotion détectée", fg="red", font=(None, 24, "bold")
            )

        self.after(33, self.update_frame)

