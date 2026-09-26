#!/usr/bin/env python3

"""
AstraMind - Person Detector

Первый модуль компьютерного зрения:
- получает изображение с камеры;
- обнаруживает человека с помощью YOLO;
- рисует рамку вокруг человека;
- показывает визуальное окно.

Пока модуль ничего не отправляет по сети.
"""

import cv2
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

CAMERA_DEVICE = 0
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

WINDOW_NAME = "AstraMind - Person Detector"


# =========================
# DETECTOR
# =========================

def main():
    print("AstraMind: loading YOLO model...")

    model = YOLO(MODEL_NAME)

    print("AstraMind: opening camera...")

    camera = cv2.VideoCapture(CAMERA_DEVICE)

    if not camera.isOpened():
        print(f"ERROR: cannot open camera {CAMERA_DEVICE}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("WARNING: camera frame is empty")
                continue

            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            person_count = 0

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[class_id]

                    if class_name != "person":
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = f"person {confidence:.2f}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    person_count += 1

            status = f"People: {person_count}"

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("AstraMind: detector stopped")


if __name__ == "__main__":
    main()