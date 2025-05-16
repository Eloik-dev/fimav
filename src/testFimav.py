from fimav.mqtt.mqtt_manager import MqttManager
from fimav.midi.midi_controller import MidiController

mqtt_manager = MqttManager()
controller = MidiController(mqtt_manager)
controller.play_midi_file(midi_file_name="jungle.mid")