import tkinter as tk
from tkinter import Canvas, Label
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
from fimav.processing.emotion_state_controller import EmotionStateController


class MainWindow(tk.Tk):
    def __init__(self, video_capture, detector, face_size, width, height):
        super().__init__()
        self.title("Face & Emotion Detection")
        self.video_capture = video_capture
        self.detector = detector
        self.emotion_controller = EmotionStateController.get_instance()
        self.face_size = face_size

        self.width = width
        self.height = height

        # Canvas for video feed
        self.canvas = Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack()
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=None)

        # Emotion label above progress bar
        self.target_label = Label(
            self, text="", font=(None, 18, "bold"), bg="black", fg="white"
        )
        # Progress bar at bottom center
        self.progress = ttk.Progressbar(
            self, orient="horizontal", length=width * 0.6, mode="determinate"
        )

        # Place label and progress bar
        self.target_label.place(relx=0.5, rely=0.9, anchor="s")
        self.progress.place(relx=0.5, rely=0.93, anchor="n")

        # Keep reference to PhotoImage
        self.photo = None

        self.prev_boxes = []
        self.smooth_factor = 0.8

        # Set interval based on desired FPS
        self.interval = 1 / 30

        # Start update loop
        self.update_frame()

    def lerp_box(self, box1, box2, t):
        return [
            int(box1[0] + (box2[0] - box1[0]) * t),
            int(box1[1] + (box2[1] - box1[1]) * t),
            int(box1[2] + (box2[2] - box1[2]) * t),
            int(box1[3] + (box2[3] - box1[3]) * t),
        ]

    def _scale_boxes(self, raw_boxes):
        scale_x = self.width / self.face_size[0]
        scale_y = self.height / self.face_size[1]
        scaled_boxes = []
        for x1, y1, x2, y2 in raw_boxes:
            scaled_x1 = max(0, int(x1 * scale_x))
            scaled_y1 = max(0, int(y1 * scale_y))
            scaled_x2 = min(self.width, int(x2 * scale_x))
            scaled_y2 = min(self.height, int(y2 * scale_y))
            scaled_boxes.append(
                (scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)
            )
        return scaled_boxes

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()
        if frame is None:
            time.sleep(self.interval)
            return self.update_frame()

        frame = cv2.resize(frame, (self.width, self.height))

        # Draw interpolated boxes
        raw_boxes = self.detector.get_latest_detection() or []
        scaled_boxes = self._scale_boxes(raw_boxes)
        interpolated_boxes = [
            (
                self.lerp_box(self.prev_boxes[i], b, self.smooth_factor)
                if i < len(self.prev_boxes)
                else b
            )
            for i, b in enumerate(scaled_boxes)
        ]
        self.prev_boxes = interpolated_boxes

        for x, y, w, h in interpolated_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update image on canvas
        data = frame[..., ::-1].tobytes()
        pil_img = Image.frombuffer(
            "RGB", (self.width, self.height), data, "raw", "RGB", 0, 1
        )
        self.photo = ImageTk.PhotoImage(pil_img, master=self.canvas)
        self.canvas.itemconfig(self.canvas_img, image=self.photo)

        # Update emotion display
        target_emotion = self.emotion_controller.get_target_emotion()
        self.target_label.config(
            text=(
                f"L'orchestre va jouer une musique {target_emotion}"
                if target_emotion
                else "Nourrissez l'orchestre de vos émotions !"
            )
        )
        self.progress["value"] = (
            self.emotion_controller.get_emotion_progress() if target_emotion else 0
        )

        time.sleep(self.interval)
        self.update_frame()
