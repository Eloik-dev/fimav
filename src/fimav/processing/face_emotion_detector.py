import cv2
import threading
import queue
import numpy as np
import ncnn
import time


class FaceEmotionDetector:
    def __init__(
        self,
        width,
        height,
        frame_queue=None,
        face_param="./models/face/ultraface_12.param",  # Default paths
        face_bin="./models/face/ultraface_12.bin",
        emo_param="./models/emotion/emotion_ferplus_12.param",
        emo_bin="./models/emotion/emotion_ferplus_12.bin",
        face_size=(320, 240),
        emo_size=(64, 64),
    ):
        # Frame I/O
        self.frame_queue = frame_queue
        self.detection_queue = queue.Queue(maxsize=1)
        self.emotion_queue = queue.Queue(maxsize=1)
        self.current_emotion = None
        self.last_bbox = None

        # 1) Load face detector
        self.face_net = ncnn.Net()
        self.face_net.load_param(face_param)
        self.face_net.load_model(face_bin)

        # 2) Load emotion classifier
        self.emo_net = ncnn.Net()
        self.emo_net.load_param(emo_param)
        self.emo_net.load_model(emo_bin)

        # Preprocess sizes
        self.face_size = face_size
        self.emo_size = emo_size
        self.width = width
        self.height = height

        # Thread control
        self.running = False
        self.face_thread = None
        self.emotion_thread = None
        self._stop_face_thread = threading.Event()
        self._stop_emotion_thread = threading.Event()

        # Emotion labels mapping
        self.emotion_labels = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]

    def start_processing(self):
        """Start the face and emotion detection thread."""
        if self.running:
            return  # Already running

        self.running = True

        self.face_thread = threading.Thread(
            target=self._face_processing_loop, daemon=True
        )
        self.face_thread.start()

        self.emotion_thread = threading.Thread(
            target=self._emotion_processing_loop, daemon=True
        )
        self.emotion_thread.start()

    def stop_processing(self):
        """Stop the face and emotion detection thread."""
        self.running = False
        self._stop_face_thread.set()  # Signal the face thread to stop
        self._stop_emotion_thread.set()  # Signal the face thread to stop
        if self.face_thread and self.face_thread.is_alive():
            self.face_thread.join()
        if self.emotion_thread and self.emotion_thread.is_alive():
            self.emotion_thread.join()
        cv2.destroyAllWindows()  # Destroy any OpenCV windows

    def get_detection_queue(self):
        return self.detection_queue

    def _face_processing_loop(self):
        print("Face detection thread started")
        while not self._stop_face_thread.is_set():
            try:
                frame = self.frame_queue.get_nowait()
                if frame is not None:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_image = cv2.resize(image_rgb, self.face_size)
                    raw_bboxes = self._detect_faces(resized_image)
                    scaled_bboxes = self._scale_boxes(raw_bboxes)

                    for raw_bbox, scaled_bbox in zip(raw_bboxes, scaled_bboxes):
                        try:
                            self.detection_queue.put_nowait((frame, scaled_bbox))
                            self.emotion_queue.put_nowait((resized_image, raw_bbox))
                            print("Face detected:", scaled_bbox)
                        except queue.Full:
                            pass
                    time.sleep(0.07)
            except queue.Empty:
                pass


    def _emotion_processing_loop(self):
        print("Emotion classification thread started")
        while not self._stop_emotion_thread.is_set():
            try:
                resized_frame, bbox = self.emotion_queue.get_nowait()
                if resized_frame is None or bbox is None:
                    continue
                emotion = self._classify_emotion(resized_frame, bbox)
                self.current_emotion = emotion
            except queue.Full:
                pass
            except queue.Empty:
                pass
            time.sleep(1)

    def get_current_emotion(self):
        return self.current_emotion

    def _scale_boxes(self, raw_boxes):
        """Scales the raw bounding box coordinates to the output display size."""
        scale_x = self.width / self.face_size[0]
        scale_y = self.height / self.face_size[1]
        scaled_boxes = []
        for x1, y1, x2, y2 in raw_boxes:
            scaled_x1 = max(0, int(x1 * scale_x))
            scaled_y1 = max(0, int(y1 * scale_y))
            scaled_x2 = min(self.width, int(x2 * scale_x))
            scaled_y2 = min(self.height, int(y2 * scale_y))
            scaled_boxes.append(
                (scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)
            )
        return scaled_boxes

    def _detect_faces(self, resized_image: np.ndarray):
        mat = ncnn.Mat.from_pixels(
            resized_image, ncnn.Mat.PixelType.PIXEL_RGB, *self.face_size
        )
        mat.substract_mean_normalize([127, 127, 127], [1.0 / 128] * 3)

        face_ex = self.face_net.create_extractor()
        face_ex.input("in0", mat)

        _, out0 = face_ex.extract("out0")
        _, out1 = face_ex.extract("out1")

        boxes = self.decode_boxes(out0, out1, score_threshold=0.8)
        return boxes

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _classify_emotion(self, frame: np.ndarray, bbox: tuple):
        x, y, x2, y2 = bbox
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

    def decode_boxes(self, scores, boxes, score_threshold=0.7, iou_threshold=0.3):
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
