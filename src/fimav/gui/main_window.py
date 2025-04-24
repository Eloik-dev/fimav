import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import threading
from fimav.processing.video_capture import VideoCapture


class MainWindow:
    def __init__(self, root, detector, face_size, width, height):
        self.root = root
        self.root.title("Video Feed with Canvas Overlay")

        # Video capture setup
        self.video_capture = VideoCapture.get_instance()
        self.width = width
        self.height = height
        self.detector = detector
        self.face_size = face_size

        # Create a Canvas to hold the video frame and overlay text
        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        # Create image item (initially empty) and text item on the Canvas
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=None)
        self.canvas_text = self.canvas.create_text(
            10,
            30,
            text="Conducting: 😊",  # initial overlay text
            fill="white",
            font=("Helvetica", 16, "bold"),
            anchor="nw",
        )

        # Streaming control
        self.interval = 1 / 30
        self.is_running = False
        self.thread = None

        # Ensure clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self):
        """Starts the video stream in a separate thread."""
        if not self.is_running:
            self.is_running = True
            self.video_capture.start_capture()
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the video stream and thread."""
        if self.is_running:
            self.is_running = False
            if self.video_capture is not None:
                self.video_capture.stop_capture()
            if self.thread is not None:
                self.thread.join()
            self.thread = None

    def _update_frame(self):
        """Fetches frames, overlays text, and updates the Canvas image."""
        while self.is_running:
            frame = self.video_capture.get_new_frame()
            if frame is None:
                print("Error: Failed to read frame. Skipping.")
                continue

            # Optionally: detect face or emotion here using self.detector
            # and update overlay text accordingly:
            # new_text = f"Emotion: {detected_emotion}"
            # self.canvas.itemconfig(self.canvas_text, text=new_text)

            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update Canvas image item
            self.canvas.itemconfig(self.canvas_img, image=img_tk)
            # Keep reference to avoid garbage collection
            self.canvas.image = img_tk

            # Throttle loop to desired FPS
            time.sleep(self.interval)

    def _on_close(self):
        """Handles window close event by stopping capture and closing."""
        self.stop()
        self.root.destroy()


# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    # pass in your detector, face_size, width, height
    app = MainWindow(root, detector=None, face_size=(128, 128), width=640, height=480)
    app.start()
    root.mainloop()
