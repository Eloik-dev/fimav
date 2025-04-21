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

        self.canvas = Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack()

        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=None)

        self.emotion_label = Label(
            self, text="", font=(None, 24, "bold"), bg="black", fg="white"
        )
        self.emotion_label.place(x=self.width // 2, y=10, anchor="n")

        self.photo = None
        self.prev_boxes = []  # <-- store previous box positions

        self.smooth_factor = 0.7

        self.after(15, self.update_frame)

    def lerp_box(self, box1, box2, t):
        """Linearly interpolate between two boxes."""
        return [
            int(box1[0] + (box2[0] - box1[0]) * t),
            int(box1[1] + (box2[1] - box1[1]) * t),
            int(box1[2] + (box2[2] - box1[2]) * t),
            int(box1[3] + (box2[3] - box1[3]) * t),
        ]

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()
        if frame is None:
            self.after(15, self.update_frame)
            return

        raw_boxes = self.detector.get_latest_detection()
        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Interpolate current boxes
        interpolated_boxes = []
        for i, new_box in enumerate(raw_boxes):
            if i < len(self.prev_boxes):
                prev_box = self.prev_boxes[i]
                interpolated = self.lerp_box(prev_box, new_box, self.smooth_factor)
            else:
                interpolated = new_box  # no previous match
            interpolated_boxes.append(interpolated)

        self.prev_boxes = interpolated_boxes  # update for next frame

        for x, y, w, h in interpolated_boxes:
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.itemconfig(self.canvas_img, image=self.photo)

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
