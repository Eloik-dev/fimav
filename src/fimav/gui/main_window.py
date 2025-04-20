import tkinter as tk
from tkinter import Canvas, Label
from PIL import Image, ImageTk
import cv2
import queue


class MainWindow(tk.Tk):
    def __init__(self, frame_queue, detector, width, height):
        super().__init__()
        self.title("Face & Emotion Detection")
        self.frame_queue = frame_queue
        self.detector = detector
        self.detection_queue = detector.get_detection_queue()
        self.target_detections = []
        self.current_detections = []
        self.lerp_alpha = 0.8

        # Setup canvas for video
        self.width = width
        self.height = height
        self.canvas = Canvas(self, width=width, height=height)
        self.canvas.pack()

        # Label for emotion text
        self.emotion_label = Label(
            self.canvas, text="", font=(None, 24, "bold"), bg="black", fg="white"
        )
        # Place the label on the canvas
        self.canvas.create_window(
            self.width // 2, 30, window=self.emotion_label, anchor="n"
        )

        # Keep reference to PhotoImage to avoid GC
        self.photo = None

        # Start update loop
        self.after(15, self.update_frame)  # ~66 FPS

    def update_frame(self):
        # Get latest frame
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if frame is None:
            self.after(15, self.update_frame)
            return

        # Sync detection boxes
        new_targets = []
        while not self.detection_queue.empty():
            try:
                _, bbox = self.detection_queue.get_nowait()
                new_targets.append(bbox)
            except queue.Empty:
                break
        if new_targets:
            self.target_detections = new_targets

        # Interpolate (lerp) boxes
        while len(self.current_detections) < len(self.target_detections):
            self.current_detections.append(
                self.target_detections[len(self.current_detections)]
            )
        while len(self.current_detections) > len(self.target_detections):
            self.current_detections.pop()
        for i, tgt in enumerate(self.target_detections):
            cur = self.current_detections[i]
            lerped = tuple(
                cur_coord + self.lerp_alpha * (tgt_coord - cur_coord)
                for cur_coord, tgt_coord in zip(cur, tgt)
            )
            self.current_detections[i] = lerped

        # First resize the frame to canvas dimensions
        frame = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw rectangles on the frame
        for x_f, y_f, w_f, h_f in self.current_detections:
            x, y, w_rec, h_rec = map(int, (x_f, y_f, w_f, h_f))
            cv2.rectangle(rgb, (x, y), (x + w_rec, y + h_rec), (0, 255, 0), 2)

        # Convert OpenCV image (BGR) to PIL image
        pil_img = Image.fromarray(rgb)

        # Convert to ImageTk
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Update emotion label
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

        # Schedule next frame
        self.after(15, self.update_frame)
