import argparse
import logging
import sys
import tkinter as tk
from fimav import __version__
from fimav.processing.video_capture import VideoCapture
from fimav.processing.face_emotion_detector import FaceEmotionDetector
from fimav.processing.emotion_state_controller import EmotionStateController
from fimav.gui.main_window import MainWindow
from fimav.mqtt.mqtt_manager import MqttManager
from fimav.midi.midi_controller import MidiController

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
    parser.add_argument("--width", type=int, default=1920, help="Initial display width")
    parser.add_argument(
        "--height", type=int, default=1080, help="Initial display height"
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Index of the camera to use"
    )
    parser.add_argument(
        "--camera-width", type=int, default=1920, help="Width of the camera to use"
    )
    parser.add_argument(
        "--camera-height", type=int, default=1080, help="Height of the camera to use"
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


def create_face_emotion_detection_thread(face_size, width, height):
    """Runs the Face Emotion Detection in a separate thread."""
    # Create the FaceEmotionDetector instance
    detector = FaceEmotionDetector(
        width,
        height,
        "models/face/ultraface_12.param",
        "models/face/ultraface_12.bin",
        "models/emotion/emotion_ferplus_12.param",
        "models/emotion/emotion_ferplus_12.bin",
        face_size,
    )
    detector.start_processing()
    print("Face Emotion Detection started in a separate thread")

    return detector


def create_gui_thread(detector, face_size, width, height):
    """
    Runs the Tkinter GUI in a separate daemon thread.

    :param width: window width
    :param height: window height
    :returns: the Thread object running the GUI
    """
    # Instantiate and run the Tkinter MainWindow
    root = tk.Tk()
    window = MainWindow(root, detector, face_size, width, height)
    window.start()
    root.mainloop()
    detector.stop_processing()
    print("GUI thread finished")


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    width = args.width
    height = args.height
    face_size = (320, 240)
    print(f"Initial display size: {width}x{height}")

    try:
        mqtt_manager = MqttManager()
        midi_controller = MidiController(mqtt_manager)

        # Create and initialize the VideoCapture instance
        VideoCapture(
            camera_index=args.camera_index,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
        
        # Create and initialize the EmotionStateController
        EmotionStateController(midi_controller)

        detector = create_face_emotion_detection_thread(
            face_size, width, height
        )
        create_gui_thread(
            detector, face_size, width, height
        )
    except KeyboardInterrupt:
        print("Ctrl-C received. Stopping...")
        if "detector" in locals():
            detector.stop_processing()
            
        VideoCapture.get_instance().stop_capture()

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
