import tkinter as tk
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys

class VideoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GStreamer Video")
        self.geometry("800x600")

        self.video_frame = tk.Frame(self, width=640, height=480)
        self.video_frame.pack(padx=10, pady=10)
        self.video_window_id = self.video_frame.winfo_id()

        Gst.init(None)
        self.pipeline = Gst.parse_launch(
            "v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! ximagesink sink-name=videosink"
        )
        self.videosink = self.pipeline.get_by_name("videosink")
        if sys.platform == 'linux':
            self.videosink.set_property("xwindow-id", self.video_window_id)

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::eos", self.on_eos)
        self.bus.connect("message::error", self.on_error)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_eos(self, bus, message):
        print("End of stream")
        self.pipeline.set_state(Gst.State.NULL)
        self.destroy()

    def on_error(self, bus, message):
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        self.pipeline.set_state(Gst.State.NULL)
        self.destroy()

    def on_close(self):
        print("Closing application")
        self.pipeline.set_state(Gst.State.NULL)
        self.destroy()

if __name__ == "__main__":
    app = VideoApp()
    app.mainloop()