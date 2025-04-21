import cv2
import threading
import numpy as np
import ncnn
import time


class FaceEmotionDetector:
    def __init__(
        self,
        width,
        height,
        video_capture,
        face_param="./models/face/ultraface_12.param",
        face_bin="./models/face/ultraface_12.bin",
        emo_param="./models/emotion/emotion_ferplus_12.param",
        emo_bin="./models/emotion/emotion_ferplus_12.bin",
        face_size=(320, 240),
        emo_size=(64, 64),
    ):
        self.width = width
        self.height = height
        self.video_capture = video_capture
        self.face_size = face_size
        self.emo_size = emo_size

        # Shared state
        self.latest_detection = []
        self.current_emotion = None
        self.shared_resized_frame = None

        # Threads
        self.running = False
        self.face_thread = None
        self.emotion_thread = None
        self._stop_face_thread = threading.Event()
        self._stop_emotion_thread = threading.Event()

        # Load models
        self.face_net = ncnn.Net()
        self.face_net.load_param(face_param)
        self.face_net.load_model(face_bin)

        self.emo_net = ncnn.Net()
        self.emo_net.load_param(emo_param)
        self.emo_net.load_model(emo_bin)

        # Emotion info
        self.emotion_labels = [
            "neutre",
            "heureux",
            "surpris",
            "triste",
            "fâché",
            "dégouté",
            "apeuré",
            "méprisant",
        ]

    def start_processing(self):
        if self.running:
            return
        self.running = True
        self._stop_face_thread.clear()
        self._stop_emotion_thread.clear()

        self.face_thread = threading.Thread(target=self._face_processing_loop)
        self.emotion_thread = threading.Thread(target=self._emotion_processing_loop, daemon=True)

        self.face_thread.start()
        self.emotion_thread.start()

    def stop_processing(self):
        self.running = False
        self._stop_face_thread.set()
        self._stop_emotion_thread.set()
        if self.face_thread and self.face_thread.is_alive():
            self.face_thread.join()
        if self.emotion_thread and self.emotion_thread.is_alive():
            self.emotion_thread.join()
        cv2.destroyAllWindows()

    def _face_processing_loop(self):
        print("Face detection thread started")
        while not self._stop_face_thread.is_set():
            frame = self.video_capture.get_latest_frame()
            if frame is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image_rgb, self.face_size)

                self.shared_resized_frame = resized_image
                self.latest_detection = self._detect_faces()

            time.sleep(0.01)

    def _emotion_processing_loop(self):
        print("Emotion classification thread started")
        while not self._stop_emotion_thread.is_set():
            frame = self.shared_resized_frame
            if frame is not None:
                self.current_emotion = self._classify_emotion(frame)
            time.sleep(0.4)

    def _detect_faces(self):
        if self.shared_resized_frame is None:
            return None

        mat = ncnn.Mat.from_pixels(
            self.shared_resized_frame, ncnn.Mat.PixelType.PIXEL_RGB, *self.face_size
        )
        mat.substract_mean_normalize([127, 127, 127], [1.0 / 128] * 3)

        ex = self.face_net.create_extractor()
        ex.input("in0", mat)
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")

        return self.decode_boxes(out0, out1, score_threshold=0.5, iou_threshold=0.2)

    def _classify_emotion(self, frame: np.ndarray):
        if self.latest_detection is None:
            return
        
        x, y, x2, y2 = self.latest_detection[0]
        w = x2 - x
        h = y2 - y
        face = frame[y : y + h, x : x + w]

        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.emo_size)

        mat = ncnn.Mat.from_pixels(
            resized, ncnn.Mat.PixelType.PIXEL_GRAY, *self.emo_size
        )

        ex = self.emo_net.create_extractor()
        ex.input("in0", mat)

        _, out = ex.extract("out0")
        scores = np.array(out)
        probs = self.softmax(scores)
        idx = int(np.argmax(probs))
        return self.emotion_labels[idx]
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def decode_boxes(self, scores, boxes, score_threshold=0.7, iou_threshold=0.2):
        """
        Convert raw outputs into actual (x, y, w, h) bounding boxes.
        """
        # Convert NCNN mats to numpy arrays
        scores_np = np.array(scores)  # shape: (4420, 2)
        boxes_np = np.array(boxes)  # shape: (4420, 4)

        # Select boxes with confidence > threshold
        face_scores = scores_np[:, 1]
        mask = face_scores > score_threshold
        filtered_scores = face_scores[mask]
        filtered_boxes = boxes_np[mask]

        # Scale boxes to absolute image size
        w, h = self.face_size
        boxes_abs = filtered_boxes.copy()
        boxes_abs[:, 0] *= w
        boxes_abs[:, 1] *= h
        boxes_abs[:, 2] *= w
        boxes_abs[:, 3] *= h

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_abs.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=score_threshold,
            nms_threshold=iou_threshold,
        )

        final_boxes = [boxes_abs[i].astype(int) for i in indices]
        return final_boxes

    def get_latest_detection(self):
        return self.latest_detection

    def get_current_emotion(self):
        return self.current_emotion
