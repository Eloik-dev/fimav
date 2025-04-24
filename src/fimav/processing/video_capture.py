import cv2

# Global OpenCV optimizations
# cv2.setNumThreads(0)  # Disable OpenCV's internal threading
# cv2.ocl.setUseOpenCL(False)  # Disable OpenCL (optional, depends on platform)
# print(cv2.getBuildInformation())


class VideoCapture:
    _instance = None

    def __new__(cls, __camera_index__=0, __camera_width__=1920, __camera_height__=1080):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        camera_index=0,
        camera_width=1920,
        camera_height=1080,
    ):
        if getattr(self, "_initialized", False):
            return

        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cap = None
        self._latest_frame = None

    @classmethod
    def get_instance(cls):
        """Return the singleton, or raise if not yet created."""
        if cls._instance is None or not getattr(cls._instance, "_initialized", False):
            raise RuntimeError("VideoCapture has not been initialized")
        return cls._instance

    def gstreamer_pipeline(self):
        return (
            f"v4l2src device=/dev/video0 ! "
            f"image/jpeg, width={self.camera_width}, height={self.camera_height}, framerate=30/1 ! "
            f"jpegdec ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink"
        )

    def start_capture(self):
        pipeline = self.gstreamer_pipeline()
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        print(
            f"Actual resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        print(f"Actual FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        return True

    def stop_capture(self):
        """
        Stops the video capture process and releases resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def get_new_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("VideoCapture: Error reading frame.")
            return None

        self._latest_frame = frame
        return frame

    def get_latest_frame(self):
        """
        Retrieves the latest captured frame atomically.
        """
        return self._latest_frame
