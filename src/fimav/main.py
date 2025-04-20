import argparse
import logging
import sys
import queue
import threading
from screeninfo import get_monitors
from fimav import __version__
from fimav.processing.video_capture import VideoCapture
from fimav.processing.face_emotion_detector import FaceEmotionDetector
from fimav.gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication

__author__ = "Eloik-dev"
__copyright__ = "Eloik-dev"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument("--width", type=int, help="Initial display width")
    parser.add_argument(
        "--height", type=int, help="Initial display height"
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_default_display_size():
    for m in get_monitors():
        if m.is_primary:
            return m.width, m.height
    return 1920, 1080


def create_face_emotion_detection_thread(frame_queue, width, height):
    """Runs the Face Emotion Detection in a separate thread."""
    # Create the FaceEmotionDetector instance
    detector = FaceEmotionDetector(
        width,
        height,
        frame_queue,
        "models/face/ultraface_12.param",
        "models/face/ultraface_12.bin",
        "models/emotion/emotion_ferplus_12.param",
        "models/emotion/emotion_ferplus_12.bin",
    )
    detector.start_processing()
    print("Face Emotion Detection started in a separate thread")
    
    return detector


def create_gui_thread(frame_queue, detector, width, height):
    """Runs the Tkinter GUI in a separate thread."""
    app = QApplication(sys.argv)
    window = MainWindow(frame_queue, detector, width, height)
    window.show()
    print("GUI thread finished")
    sys.exit(app.exec())


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    default_width, default_height = get_default_display_size()
    width = args.width if hasattr(args, "width") and args.width else default_width
    height = args.height if hasattr(args, "height") and args.height else default_height
    print(f"Initial display size: {width}x{height}")

    frame_queue = queue.Queue(maxsize=1)
    video_capture = VideoCapture(
        camera_index=0,
        frame_queue=frame_queue,
    )

    if video_capture.start_capture():
        detector = create_face_emotion_detection_thread(frame_queue, width, height)        
        create_gui_thread(frame_queue, detector, width, height)
    else:
        _logger.error("Failed to start video capture.")

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
