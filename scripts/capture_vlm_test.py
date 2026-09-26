#!/usr/bin/env python3

import time
import cv2


CAMERA_DEVICE = 0
OUTPUT_FILE = "/tmp/astramind_vlm_test.jpg"


def main():
    print("Opening Astra Pro...")

    camera = cv2.VideoCapture(CAMERA_DEVICE)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_DEVICE}"
        )

    # Даём камере несколько кадров для стабилизации.
    frame = None

    for _ in range(10):
        success, current_frame = camera.read()

        if success:
            frame = current_frame

        time.sleep(0.05)

    camera.release()

    if frame is None:
        raise RuntimeError("No frame received from camera")

    if not cv2.imwrite(OUTPUT_FILE, frame):
        raise RuntimeError("Failed to save image")

    height, width = frame.shape[:2]

    print("Frame saved:")
    print(OUTPUT_FILE)
    print(f"Resolution: {width} x {height}")


if __name__ == "__main__":
    main()