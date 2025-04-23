import threading
import cv2

# Global OpenCV optimizations
cv2.setNumThreads(0)  # Disable OpenCV's internal threading
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL (optional, depends on platform)
print(cv2.getBuildInformation())


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
        self._latest_frame = None

    def start_capture(self):
        """
        Starts the video capture process in a separate thread.
        """
        gst_pipeline = (
            "v4l2src device=/dev/video{0} io-mode=4 ! "
            "video/x-raw,format=YUY2,width={1},height={2},framerate=30/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "          # drop old frames
            "glupload ! "                                          # upload to GPU
            "glcolorconvert ! "                                    # GPU colorspace convert
            "video/x-raw(memory:GLMemory),format=RGBA,width={1},height={2} ! "
            "glimagesink sync=false"                               # render via OpenGL/KMS
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
            self._latest_frame = frame

    def get_latest_frame(self):
        """
        Retrieves the latest captured frame atomically.
        """
        return self._latest_frame
