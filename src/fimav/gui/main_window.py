from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import QTimer, Qt
import cv2
import queue

class MainWindow(QWidget):
    def __init__(self, frame_queue, detector, width, height):
        super().__init__()
        self.setWindowTitle("Face & Emotion Detection")

        self.frame_queue = frame_queue
        self.detector = detector
        self.detection_queue = detector.get_detection_queue()

        self.target_detections = []
        self.current_detections = []
        self.lerp_alpha = 0.2

        self.label = QLabel()
        self.emotion_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.emotion_label)
        self.setLayout(layout)

        self.width = width
        self.height = height
        self.setFixedSize(width, height)
        self.showFullScreen()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)  # ~66 FPS

    def update_frame(self):
        # Always take the latest frame
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if frame is None:
            return

        # Sync detection boxes
        new_targets = []
        while not self.detection_queue.empty():
            try:
                _, bbox = self.detection_queue.get_nowait()
                new_targets.append(bbox)
            except queue.Empty:
                break
        if new_targets:
            self.target_detections = new_targets

        # Interpolate (lerp) boxes
        while len(self.current_detections) < len(self.target_detections):
            self.current_detections.append(self.target_detections[len(self.current_detections)])
        while len(self.current_detections) > len(self.target_detections):
            self.current_detections.pop()
        for i, tgt in enumerate(self.target_detections):
            cur = self.current_detections[i]
            lerped = tuple(cur_coord + self.lerp_alpha * (tgt_coord - cur_coord)
                        for cur_coord, tgt_coord in zip(cur, tgt))
            self.current_detections[i] = lerped

        # Convert and display image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        # Scale pixmap to fit the screen
        pixmap = pixmap.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)

        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for x_f, y_f, w_f, h_f in self.current_detections:
            x, y, w_rec, h_rec = map(int, (x_f, y_f, w_f, h_f))
            painter.drawRect(x, y, w_rec, h_rec)
        painter.end()

        self.label.setPixmap(pixmap)

        # Emotion display
        current_emotion = self.detector.get_current_emotion()
        if current_emotion:
            self.emotion_label.setStyleSheet("font-weight: bold; font-size: 24pt; color: green;")
            self.emotion_label.setText(f"<p align='center'>Vous êtes: <span style='font-style: italic'>{current_emotion}</span></p>")
        else:
            self.emotion_label.setStyleSheet("font-weight: bold; font-size: 24pt; color: red;")
            self.emotion_label.setText("<p align='center'>Aucune émotion détectée</p>")

