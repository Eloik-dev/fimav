from mido import MidiFile
import threading
import os
import json

"""
    Classe pour jouer des fichiers MIDI sur le broker MQTT
"""
class MidiController:
    def __init__(self, mqtt_manager):
        self.mqtt_manager = mqtt_manager
        self.midi_thread = None
        self._stop_event = threading.Event()
        self.lock = threading.Lock()

    def play_midi_file(self, midi_file_name):
        file_path = f"midi/{midi_file_name}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with self.lock:
            # Arrêter la lecture si elle est en cours
            self._stop_event.set()
            if self.midi_thread and self.midi_thread.is_alive():
                self.midi_thread.join()

            # Lancer ou relancer la lecture
            self._stop_event.clear()
            self.midi_thread = threading.Thread(
                target=self._play_midi_file, args=(file_path,)
            )
            self.midi_thread.start()

    def _play_midi_file(self, file_path):
        print(f"Playing MIDI: {file_path}")
        try:
            midi_file = MidiFile(file_path)
            for message in midi_file.play():
                if self._stop_event.is_set():
                    print("Playback interrupted.")
                    break

                # Build custom payload
                if message.type in ('note_on', 'note_off'):
                    payload = {
                        'event': 0 if message.type == 'note_on' else 1,
                        'channel': message.channel,
                        'note': message.note,
                        'velocity': message.velocity
                    }
                    topic = 'orchestre'
                    
                    print(f"Envoi de la note: {payload}")
                    
                    # Send JSON string to MQTT
                    self.mqtt_manager.send_midi(json.dumps(payload))
                else:
                    # Optionally handle other message types or ignore
                    continue
        finally:
            print("Playback finished or stopped.")

    def stop(self):
        with self.lock:
            self._stop_event.set()
            if self.midi_thread and self.midi_thread.is_alive():
                self.midi_thread.join()
                
    def is_playing(self):
        return self.midi_thread and self.midi_thread.is_alive()
