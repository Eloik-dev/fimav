import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time


class CameraTest:
    def __init__(self, root, camera_index=0):
        """
        Initializes the CameraApp.

        Args:
            root (tk.Tk): The main Tkinter window.
            camera_index (int, optional): The index of the camera to use. Defaults to 0.
        """
        self.root = root
        self.root.title("Camera Stream")
        self.camera_index = camera_index
        self.cap = None  # Initialize capture object
        self.video_frame = tk.Label(root)
        self.video_frame.pack()
        self.is_running = False
        self.thread = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close

    def start_stream(self):
        """Starts the video stream in a separate thread."""
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(
                    f"Error: Could not open camera {self.camera_index}.  Check if the camera is plugged in and the index is correct."
                )
                return  # Exit if camera cannot be opened

            self.is_running = True
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.start()

    def stop_stream(self):
        """Stops the video stream."""
        if self.is_running:
            self.is_running = False
            if self.cap is not None:  # Check if camera was ever opened
                self.cap.release()
            if self.thread is not None:  # Check if thread was ever started
                self.thread.join()  # Wait for thread to finish
            self.cap = None  # reset
            self.thread = None

    def update_frame(self):
        """Reads frames from the camera and updates the Tkinter label."""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame.  Stopping stream.")
                    self.stop_stream()  # Stop stream if reading fails
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.LANCZOS)  # Resize for display
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk  # Keep a reference!
            except Exception as e:
                print(f"Error in update_frame: {e}")
                self.stop_stream()  # stop stream on error
                return
            time.sleep(0.01)  # Add a small delay

    def on_close(self):
        """Handles window close event."""
        self.stop_stream()  # Stop the stream before exiting
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraTest(root, camera_index=0)  # Use camera index 0 (default webcam)
    app.start_stream()
    root.mainloop()

