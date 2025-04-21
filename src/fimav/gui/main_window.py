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

        self.width = width
        self.height = height

        # Setup canvas and image container
        self.canvas = Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack()

        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=None)

        # Emotion label
        self.emotion_label = Label(
            self, text="", font=(None, 24, "bold"), bg="black", fg="white"
        )
        self.emotion_label.place(x=self.width // 2, y=10, anchor="n")

        self.photo = None  # keep reference to avoid garbage collection

        self.after(15, self.update_frame)  # ~60 FPS; adapt as needed

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()
        if frame is None:
            self.after(15, self.update_frame)
            return

        current_detections = self.detector.get_latest_detection()
        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw face boxes
        for x, y, w, h in current_detections:
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert to Tkinter image
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=pil_img)

        # Efficient canvas update
        self.canvas.itemconfig(self.canvas_img, image=self.photo)

        # Emotion label update
        current_emotion = self.detector.get_current_emotion()
        if current_emotion:
            self.emotion_label.config(
                text=f"Vous êtes {current_emotion}",
                fg="green",
                font=(None, 24, "bold", "italic"),
            )
        else:
            self.emotion_label.config(
                text="Aucune émotion détectée", fg="red", font=(None, 24, "bold")
            )

        self.after(15, self.update_frame)
