import cv2
import threading
import queue


class VideoCapture:
    def __init__(
        self,
        camera_index=0,
        frame_queue=None,
        camera_width=1920,
        camera_height=1080,
    ):
        """
        Initializes the VideoCapture object.

        Args:
            camera_index (int): Index of the camera to use.
            frame_queue (queue.Queue, optional): Queue to store captured frames.
                If None, a new queue is created.
            capture_width (int, optional): Width of the captured frames.
            capture_height (int, optional): Height of the captured frames.
            buffer_size (int): Maximum number of frames to buffer.
        """
        self.camera_index = camera_index
        self.frame_queue = frame_queue
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cap = None
        self.running = False
        self.capture_thread = None

    def start_capture(self):
        """
        Starts the video capture process in a separate thread.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.running = False
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Actual camera resolution: {actual_width} x {actual_height}")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Actual FPS: {fps}")

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True  # Allow the main program to exit
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

            # Always drop stale frames and keep only the latest
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Drop one
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Shouldn't happen due to pre-drop, but just in case
