import paho.mqtt.client as mqtt

"""
    Read MIDI files, 
"""

class MqttManager:
    """Simple class to manage MQTT communication."""

    def __init__(self):
        """Initialize the MQTT manager."""
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.username_pw_set("orchestrateur", "Orchestrateur1234")
        self._client.connect("localhost", 1884)
        self._client.loop_start()
        self._topic_out = "fimav/orchestre"

    def _on_connect(self, __client__, __userdata__, __flags__, rc):
        """Callback when the client is connected."""
        print("Connected to MQTT broker with result code " + str(rc))

    def _on_disconnect(self, __client__, __userdata__, rc):
        """Callback when the client is disconnected."""
        print("Disconnected from MQTT broker with result code " + str(rc))

    def send_midi(self, msg):
        """Send a MIDI message as a string to the MQTT broker."""
        self._client.publish(self._topic_out, str(msg))  
