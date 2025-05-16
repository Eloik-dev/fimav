import time
import random

"""
    Classe pour la gestion de l'emotion de l'orchestre
"""
class EmotionStateController:
    _instance = None
    DELAY = 1.5
    
    # Modifier avec des musiques neutres
    neutral_songs = [
        "neutre_boucle_1.mid",
        "neutre_boucle_2.mid",
    ]

    # Modifier avec des musiques joyeuses
    # happy_songs = [
    #     "aventureux.mid",
    #     "intrigant.mid",
    #     "joyeux__la_penta_majeur_1.mid",
    #     "jungle.mid",
    # ]
    happy_songs = ["Test.mid"]
    
    # Modifier avec des musiques surprenantes
    surprised_songs = [
        "alarmant_v2.mid",
    ]

    # Modifier avec des musiques tristes
    sad_songs = [
        "hypnotique_v2.mid",
    ]

    # Libellés des émotions
    emotion_labels = [
        "neutre",
        "heureuse",
        "surprenante",
        "triste",
        "enrageante",
        "dégoutante",
        "apeurante",
        "méprisante",
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
        self.current_neutral_song = None
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Return the singleton, or raise if not yet created."""
        if cls._instance is None or not getattr(cls._instance, "_initialized", False):
            raise RuntimeError("EmotionStateController has not been initialized")
        return cls._instance

    def update_emotion(self, emotion_idx: int):
        # Remettre la dernière émotion en cours à null si elle est terminée
        if self.last_emotion is not None and not self.midi.is_playing(): 
            self.last_emotion = None 
        
        # Ne rien faire si l'émotion est neutre
        if emotion_idx == 0:
            self.target_emotion = None
            self.emotion_start_time = None
            if self.current_neutral_song is None:
                self.current_neutral_song = random.choice(self.neutral_songs)
                self.midi.play_midi_file(self.current_neutral_song)
            return

        # Ignorer si la même émotion est en cours
        if self.midi.is_playing() and emotion_idx == self.last_emotion:
            return

        now = time.time()

        # Changer d'émotion si elle est différente
        if emotion_idx != self.target_emotion:
            self.target_emotion = emotion_idx
            self.emotion_start_time = now
            return

        # Changer de musique si le temps est suffisant
        if now - (self.emotion_start_time or now) >= self.DELAY:
            self._trigger_song(emotion_idx)
            self.target_emotion = None
            self.emotion_start_time = None

    def get_target_emotion(self) -> int:
        return self.target_emotion
    
    def get_last_emotion(self) -> int:
        return self.last_emotion
    
    def get_last_emotion_string(self) -> str:
        if self.last_emotion is None:
            return self.emotion_labels[0]
        return self.emotion_labels[self.last_emotion]
    
    def get_emotion_labels(self) -> list:
        return self.emotion_labels

    def get_emotion_progress(self) -> float:
        if not self.target_emotion or not self.emotion_start_time:
            return 0.0
        elapsed = time.time() - self.emotion_start_time
        return min(elapsed / self.DELAY, 1)

    def _trigger_song(self, emotion_idx: int):
        if emotion_idx == 1:
            midi = random.choice(self.happy_songs)
        elif emotion_idx == 3:
            midi = random.choice(self.sad_songs)
        else:
            return
        self.midi.play_midi_file(midi)
        self.last_emotion = emotion_idx

