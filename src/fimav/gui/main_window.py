import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time


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

        self.interval = round((1 / 30) * 1000)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        while self.video_capture.is_running():
            try:
                frame = self.video_capture.get_latest_frame()
                if frame is None:
                    print("Error: Failed to read frame.  Retrying...")
                    time.sleep(0.1)
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
                img = Image.fromarray(frame)
                img = img.resize((1920, 1080), Image.LANCZOS)  # Resize for display
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk  # Keep a reference!
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.on_close()  # stop stream on error
                return
            time.sleep(0.01)  # Add a small delay
        
    def on_close(self):
        """Handles window close event."""
        self.video_capture.stop()  # Stop the stream before exiting
        self.root.destroy()