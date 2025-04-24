import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from fimav.processing.video_capture import VideoCapture
from fimav.processing.face_emotion_detector import FaceEmotionDetector
from fimav.processing.emotion_state_controller import EmotionStateController


class MainWindow:
    def __init__(self, root, face_size, width, height):
        self.root = root
        self.root.title("Video Feed with Progress Bar Overlay")

        # Video capture setup
        self.video_capture = VideoCapture.get_instance()
        self.detector = FaceEmotionDetector.get_instance()
        self.emotion_controller = EmotionStateController.get_instance()
        self.width = width
        self.height = height
        self.face_size = face_size

        # Create a Canvas to hold the video frame
        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        # Create image item (initially empty)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=None)

        # Streaming control
        self.interval = 1 / 30
        self.is_running = False
        self.thread = None

        self.no_emotion_text_image = self.render_text_image(
            "Contrôlez l'orchestre avec vos émotions !", "Arial", 20
        )

        base_emotion_text = "La prochaine musique sera "
        self.emotions_with_fonts = [
            self.render_text_image(base_emotion_text + "heureuse", "Arial", 20),
            self.render_text_image(base_emotion_text + "surprenante", "Arial", 20),
            self.render_text_image(base_emotion_text + "triste", "Arial", 20),
            self.render_text_image(base_emotion_text + "enrageante", "Arial", 20),
            self.render_text_image(base_emotion_text + "dégoutante", "Arial", 20),
            self.render_text_image(base_emotion_text + "apeurante", "Arial", 20),
            self.render_text_image(base_emotion_text + "méprisante", "Arial", 20),
        ]

        # Ensure clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self):
        """Starts the video stream in a separate thread."""
        if not self.is_running:
            self.is_running = True
            self.video_capture.start_capture()
            self.detector.start_processing()

            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the video stream and thread."""
        if self.is_running:
            self.is_running = False
            if self.video_capture is not None:
                self.video_capture.stop_capture()
                self.detector.stop_processing()
            if self.thread is not None:
                self.thread.join()
            self.thread = None

    def _update_frame(self):
        """Fetches frames, overlays text and progress bar with OpenCV, updates the Canvas image."""
        while self.is_running:
            frame = self.video_capture.get_new_frame()
            if frame is None:
                print("Error: Failed to read frame. Skipping.")
                continue

            raw_boxes = self.detector.get_latest_detection() or []
            scaled_boxes = self._scale_boxes(raw_boxes)

            # Draw boxes
            for x, y, w, h in scaled_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw progress bar at bottom middle
            bar_width = int(self.width * 0.6)
            bar_height = 20
            bar_x = int((self.width - bar_width) / 2)
            bar_y = self.height - 40
            filled_width = int(
                bar_width * self.emotion_controller.get_emotion_progress()
            )

            # Background bar (dark grey)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                cv2.FILLED,
            )
            # Filled portion (green)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + filled_width, bar_y + bar_height),
                (0, 255, 0),
                cv2.FILLED,
            )
            # Border (white)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2,
            )

            # Show current emotion above progress bar
            current_emotion = self.emotion_controller.get_target_emotion()
            if current_emotion is None:
                text_image = self.no_emotion_text_image
            else:
                text_image = self.emotions_with_fonts[current_emotion]
            
            h, w, _ = text_image
            x = bar_x + int((bar_width - w) / 2)
            y = bar_y - 40
            frame[y:y+h, x:x+w] = text_image

            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update Canvas image item
            self.canvas.itemconfig(self.canvas_img, image=img_tk)
            # Keep reference to avoid garbage collection
            self.canvas.image = img_tk

            # Throttle loop
            time.sleep(self.interval)

    def render_text_image(self, text, font_path="DejaVuSans.ttf", font_size=32):
        font = ImageFont.truetype(font_path, font_size)
        text_size = font.getsize(text)
        img = Image.new("RGB", text_size, (0, 0, 0))  # Transparent background if needed
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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

    def _on_close(self):
        """Handles window close event by stopping capture and closing."""
        self.stop()
        self.root.destroy()
