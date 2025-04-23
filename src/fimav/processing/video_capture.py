import cv2
import threading
import queue


class VideoCapture:
    def __init__(
        self,
        camera_index=0,
        camera_width=1920,
        camera_height=1080,
    ):
        """
        Initializes the VideoCapture object.

        Args:
            camera_index (int): Index of the camera to use.
            capture_width (int, optional): Width of the captured frames.
            capture_height (int, optional): Height of the captured frames.
            buffer_size (int): Maximum number of frames to buffer.
        """
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cap = None
        self.running = False
        self.capture_thread = None

        self.latest_frame = None

    def start_capture(self):
        """
        Starts the video capture process in a separate thread.
        """
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.running = False
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        print(
            f"Actual resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        print(f"Actual FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        return True

    def stop_capture(self):
        """
        Stops the video capture process and releases resources.
        """
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None  # Ensure cap is reset after release

    def _capture_loop(self):
        while self.running:
            # Read the latest frame
            ret, frame = self.cap.read()
            if not ret:
                print("VideoCapture: Error reading frame. Stopping capture.")
                self.running = False
                break

            self.latest_frame = frame

    def get_latest_frame(self):
        return self.latest_frame
