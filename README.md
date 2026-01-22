# PDIp2p3
opencv-python
tensorflow
numpy
deep-sort-realtime
import tensorflow as tf
import numpy as np

class VehicleDetector:
    def __init__(self):
        self.model = tf.saved_model.load(
            "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        )

        self.vehicle_classes = [2, 3, 5, 7]  
        # 2: carro, 3: moto, 5: ônibus, 7: caminhão

    def detect(self, frame):
        img = tf.convert_to_tensor(frame)
        img = img[tf.newaxis, ...]

        detections = self.model(img)

        boxes = detections["detection_boxes"][0].numpy()
        scores = detections["detection_scores"][0].numpy()
        classes = detections["detection_classes"][0].numpy().astype(int)

        results = []

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5 and cls in self.vehicle_classes:
                results.append((box, score))

        return results
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame_shape):
        h, w, _ = frame_shape
        formatted = []

        for box, score in detections:
            y1, x1, y2, x2 = box
            formatted.append([
                x1 * w,
                y1 * h,
                (x2 - x1) * w,
                (y2 - y1) * h,
                score
            ])

        tracks = self.tracker.update_tracks(formatted)
        return tracks
def estimate_congestion(vehicle_count):
    if vehicle_count < 10:
        return "Baixo"
    elif vehicle_count < 25:
        return "Moderado"
    else:
        return "Alto"
import cv2
from detector import VehicleDetector
from tracker import VehicleTracker
from congestion import estimate_congestion

video_path = "video_rua.mp4"  # vídeo de entrada

detector = VehicleDetector()
tracker = VehicleTracker()

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame.shape)

    vehicle_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        vehicle_count += 1
        x, y, w, h = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    congestion_level = estimate_congestion(vehicle_count)

    cv2.putText(
        frame,
        f"Veiculos: {vehicle_count} | Congestionamento: {congestion_level}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Monitoramento de Trafego", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
