import time
import random


class EmotionStateController:
    _instance = None
    DELAY = 1.5

    # Modifier avec des musiques joyeuses
    happy_songs = [
        "Test.mid",
    ]

    # Modifier avec des musiques tristes
    sad_songs = [
        "Test.mid",
    ]

    def __new__(cls, __midi_controller__=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, midi_controller=None):
        if getattr(self, "_initialized", False):
            return
        if midi_controller is None:
            raise ValueError("Must initialize with arguments first")

        self.midi = midi_controller
        self.emotion_start_time = None
        self.last_emotion = None
        self.target_emotion = None
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Return the singleton, or raise if not yet created."""
        if cls._instance is None or not getattr(cls._instance, "_initialized", False):
            raise RuntimeError("EmotionStateController has not been initialized")
        return cls._instance

    def update_emotion(self, emotion: str):
        # reset on neutral
        if emotion == "neutre":
            self.target_emotion = None
            self.emotion_start_time = None
            return

        # ignore same as current song
        if self.midi.is_playing() and emotion == self.last_emotion:
            return

        now = time.time()

        # new hold cycle
        if emotion != self.target_emotion:
            self.target_emotion = emotion
            self.emotion_start_time = now
            return

        # held long enough?
        if now - (self.emotion_start_time or now) >= self.DELAY:
            self._trigger_song(emotion)
            self.target_emotion = None
            self.emotion_start_time = None

    def reset_last_emotion(self):
        self.last_emotion = None

    def get_target_emotion(self) -> str:
        return self.target_emotion

    def get_emotion_progress(self) -> float:
        if not self.target_emotion or not self.emotion_start_time:
            return 0.0
        elapsed = time.time() - self.emotion_start_time
        return min(elapsed / self.DELAY, 1)

    def _trigger_song(self, emotion: str):
        if emotion == "heureuse":
            midi = random.choice(self.happy_songs)
        elif emotion == "triste":
            midi = random.choice(self.sad_songs)
        else:
            return
        self.midi.play_midi_file(midi)
        self.last_emotion = emotion
