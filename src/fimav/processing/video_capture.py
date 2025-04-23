import threading
import cv2

# Global OpenCV optimizations
cv2.setNumThreads(0)  # Disable OpenCV's internal threading
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL (optional, depends on platform)

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
            capture_width (int): Width of the captured frames.
            capture_height (int): Height of the captured frames.
        """
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cap = None
        self.running = False
        self.capture_thread = None

        self._lock = threading.Lock()  # Light lock for thread-safe frame access
        self._frame_slot_0 = None
        self._frame_slot_1 = None
        self._active_index = 0  # Atomic-like swap index

    def start_capture(self):
        """
        Starts the video capture process in a separate thread.
        """
        gst_pipeline = (
            "v4l2src device=/dev/video{} ! "
            "image/jpeg,framerate=30/1,width={},height={} ! "
            "jpegdec ! "
            "videoconvert ! "
            "appsink drop=true max-buffers=1"
        ).format(
            self.camera_index,
            self.camera_width,
            self.camera_height
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

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
            self.cap = None


    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("VideoCapture: Error reading frame.")
                self.running = False
                break

            with self._lock:
                if self._active_index == 0:
                    self._frame_slot_1 = frame
                    self._active_index = 1
                else:
                    self._frame_slot_0 = frame
                    self._active_index = 0

    def get_latest_frame(self):
        """
        Retrieves the latest captured frame in a thread-safe manner.
        """
        with self._lock:
            if self._active_index == 0:
                return self._frame_slot_0
            else:
                return self._frame_slot_1
