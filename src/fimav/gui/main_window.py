import tkinter as tk
from tkinter import Canvas, Label
from PIL import Image, ImageTk
import cv2
import queue


class MainWindow(tk.Tk):
    def __init__(self, video_capture, detector, width, height):
        super().__init__()
        self.title("Face & Emotion Detection")
        self.video_capture = video_capture
        self.detector = detector
        self.current_detections = []

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

        # Previous box positions for lerping
        self.prev_rects = {}
        self.lerp_speed = 0.5

        # Start update loop
        self.after(15, self.update_frame)

    def update_frame(self):
        frame = self.video_capture.get_latest_frame()

        if frame is None:
            self.after(15, self.update_frame)
            return

        self.current_detections = self.detector.get_latest_detection()

        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Interpolate and draw rectangles
        new_rects = []
        for idx, (x, y, w, h) in enumerate(self.current_detections):
            prev = self.prev_rects.get(idx, (x, y, w, h))

            # Linear interpolation
            lerped_x = int(prev[0] + (x - prev[0]) * self.lerp_speed)
            lerped_y = int(prev[1] + (y - prev[1]) * self.lerp_speed)
            lerped_w = int(prev[2] + (w - prev[2]) * self.lerp_speed)
            lerped_h = int(prev[3] + (h - prev[3]) * self.lerp_speed)

            new_rects.append((lerped_x, lerped_y, lerped_w, lerped_h))
            self.prev_rects[idx] = (lerped_x, lerped_y, lerped_w, lerped_h)

            # Draw
            cv2.rectangle(
                rgb,
                (lerped_x, lerped_y),
                (lerped_x + lerped_w, lerped_y + lerped_h),
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

        self.after(15, self.update_frame)
