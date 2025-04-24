import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import threading


class MainWindow:
    def __init__(self, root, video_capture, detector, face_size, width, height):
        self.root = root
        self.root.title("Video Feed")
        self.video_capture = video_capture
        self.width = width
        self.height = height
        self.detector = detector
        self.face_size = face_size

        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.interval = 1 / 30

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def start(self):
        """Starts the video stream in a separate thread using GStreamer."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._update_frame)
            self.thread.start()

    def stop(self):
        """Stops the video stream."""
        if self.is_running:
            self.is_running = False
            if self.video_capture is not None:  # Check if camera was ever opened
                self.video_capture.stop_capture()
            if self.thread is not None:  # Check if thread was ever started
                self.thread.join()  # Wait for thread to finish
            self.thread = None

    def _update_frame(self):
        """Reads frames from the camera and updates the Tkinter label."""
        while self.is_running:
            try:
                frame = self.video_capture.get_latest_frame()
                if not frame:
                    print("Error: Failed to read frame. Skipping.")
                    continue
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
                img = Image.fromarray(frame)
                img = img.resize((1920, 1080), Image.LANCZOS)  # Resize for display
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk  # Keep a reference!
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.stop()  # stop stream on error
                return
            time.sleep(self.interval)  # Add a small delay

    def _on_close(self):
        """Handles window close event."""
        self.stop()  # Stop the stream before exiting
        self.root.destroy()