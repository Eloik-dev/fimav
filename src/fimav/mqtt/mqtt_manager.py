import mido
import paho.mqtt.client as mqtt


class MqttManager:
    """Simple class to manage MQTT communication."""

    def __init__(self, host, port, topic_in, topic_out):
        """Initialize the MQTT manager."""
        self._client = mqtt.Client()
        self._client.connect(host, port)
        self._topic_in = topic_in
        self._topic_out = topic_out

    def send_midi(self, msg):
        """Send a MIDI message as a string to the MQTT broker."""
        self._client.publish(self._topic_out, msg.bytes().hex())