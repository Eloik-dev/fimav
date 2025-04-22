from mido import MidiFile
import threading
import time
import os

class MidiController:
    def __init__(self, mqtt_manager):
        self.mqtt_manager = mqtt_manager
        self.midi_file = None

        self.midi_thread = None
        self._stop_midi_thread = threading.Event()
        self.running = False

    def start_processing(self):
        if self.running:
            return

        self.running = True
        self._stop_midi_thread.clear()

        self.midi_thread = threading.Thread(target=self._run, daemon=True)
        self.midi_thread.start()

    def stop_processing(self):
        self.running = False
        self._stop_midi_thread.set()

        if self.midi_thread and self.midi_thread.is_alive():
            self.midi_thread.join()

    def _run(self):
        while not self._stop_midi_thread.is_set():
            if self.midi_file:
                self._process_midi()
            else:
                print("No MIDI file selected.")
                time.sleep(1)

    def set_midi_file_from_path(self, midi_file_name):
        file_path = "midi/{}".format(midi_file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.midi_file = MidiFile(file_path)

    def _process_midi(self):
        print("Starting MIDI processing...")
        try:
            for message in self.midi_file.play():
                if self._stop_midi_thread.is_set():
                    break
                self.mqtt_manager.send_midi(message)
                print(message)
        finally:
            print("Stopping MIDI processing...")

