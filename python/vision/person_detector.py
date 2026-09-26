#!/usr/bin/env python3

"""
AstraMind - Person Detector

Получает изображение с камеры, обнаруживает людей с помощью YOLO
и отправляет последнее актуальное событие на MacBook.

Pipeline:

Astra Pro
    ↓
OpenCV
    ↓
YOLO
    ↓
стабилизация состояния
    ↓
pending event
    ↓
MacSpeaker
    ↓
SSH
    ↓
MacBook
    ↓
say
"""


import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CAMERA_DEVICE = 0

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45

WINDOW_NAME = "AstraMind - Person Detector"

# Сколько секунд новое состояние должно сохраняться,
# чтобы мы считали его настоящим изменением.
STATE_STABLE_SECS = 0.5

# Минимальная пауза между двумя сообщениями.
AUDIO_COOLDOWN_SECS = 1.0

# SSH alias из ~/.ssh/config
SEND_HOST = "astramind-mac"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEND_TEXT_SCRIPT = (
    PROJECT_ROOT
    / "python"
    / "common"
    / "send_text_to_mac.py"
)


# ============================================================
# MAC SPEAKER
# ============================================================

class MacSpeaker:
    """
    Отвечает за отправку текста на Mac.

    В каждый момент времени хранится только одно pending-событие.
    Если во время ожидания приходит новое событие, старое заменяется.

    Благодаря одному worker-потоку два сообщения не могут
    одновременно запускать say/afplay на Mac.
    """

    def __init__(self, host: str, script: Path):
        self.host = host
        self.script = script

        self.pending_text = None

        # Время окончания последнего воспроизведения.
        self.last_finished_time = 0.0

        self.condition = threading.Condition()

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.worker.start()

    def submit(self, text: str) -> None:
        """
        Добавляет новое сообщение.

        Если предыдущее сообщение ещё ожидает отправки,
        оно заменяется новым.
        """

        with self.condition:
            self.pending_text = text
            self.condition.notify()

    def _worker(self) -> None:
        """
        Фоновый worker.

        Последовательно:
        1. ждёт событие;
        2. ждёт окончания cooldown;
        3. отправляет текст на Mac;
        4. ждёт завершения send_text_to_mac.py;
        5. принимает следующее актуальное событие.
        """

        while True:

            with self.condition:

                while self.pending_text is None:
                    self.condition.wait()

                now = time.monotonic()

                cooldown_remaining = (
                    AUDIO_COOLDOWN_SECS
                    - (now - self.last_finished_time)
                )

                if cooldown_remaining > 0:
                    self.condition.wait(
                        timeout=cooldown_remaining
                    )

                    continue

                # Забираем последнее актуальное событие.
                text = self.pending_text
                self.pending_text = None

            self._send(text)

            self.last_finished_time = time.monotonic()

    def _send(self, text: str) -> None:
        """
        Запускает send_text_to_mac.py.

        subprocess.run используется намеренно:
        worker ждёт, пока Mac закончит воспроизведение.
        Сам детектор при этом продолжает работать,
        потому что worker находится в отдельном потоке.
        """

        command = [
            sys.executable,
            str(self.script),
            "--text",
            text,
            "--host",
            self.host,
        ]

        print(f"[MAC] Sending: {text}")

        try:
            result = subprocess.run(
                command,
                check=False,
            )

            if result.returncode == 0:
                print("[MAC] Playback finished.")

            else:
                print(
                    f"[MAC] send_text_to_mac.py "
                    f"returned code {result.returncode}"
                )

        except Exception as error:
            print(f"[MAC] Send error: {error}")


# ============================================================
# DRAWING
# ============================================================

def draw_person_box(
    frame,
    box,
    confidence: float,
) -> None:

    x1, y1, x2, y2 = map(
        int,
        box,
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


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Проверяем helper
    # --------------------------------------------------------

    if not SEND_TEXT_SCRIPT.exists():

        print(
            "ERROR: send_text_to_mac.py not found:"
        )

        print(SEND_TEXT_SCRIPT)

        return 1

    # --------------------------------------------------------
    # Загружаем YOLO
    # --------------------------------------------------------

    print(
        "AstraMind: loading YOLO model..."
    )

    model = YOLO(MODEL_NAME)

    print(
        "AstraMind: YOLO model loaded."
    )

    # --------------------------------------------------------
    # Проверяем CUDA
    # --------------------------------------------------------

    try:

        import torch

        print(
            "Torch:",
            torch.__version__,
        )

        print(
            "CUDA available:",
            torch.cuda.is_available(),
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

    except Exception as error:

        print(
            f"Could not read CUDA information: {error}"
        )

    # --------------------------------------------------------
    # Открываем камеру
    # --------------------------------------------------------

    print(
        f"AstraMind: opening camera "
        f"{CAMERA_DEVICE}..."
    )

    camera = cv2.VideoCapture(
        CAMERA_DEVICE
    )

    if not camera.isOpened():

        print(
            f"ERROR: cannot open camera "
            f"{CAMERA_DEVICE}"
        )

        return 1

    print(
        "AstraMind: camera opened."
    )

    # --------------------------------------------------------
    # Создаём окно
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        960,
        540,
    )

    # --------------------------------------------------------
    # Mac speaker
    # --------------------------------------------------------

    speaker = MacSpeaker(
        host=SEND_HOST,
        script=SEND_TEXT_SCRIPT,
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    # Состояние, которое YOLO предлагает прямо сейчас.
    candidate_person_present = False

    # Момент начала текущего candidate state.
    candidate_since = time.monotonic()

    # Состояние, которое мы уже считаем подтверждённым.
    stable_person_present = False

    print(
        "AstraMind: detector started."
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Получаем кадр
            # ------------------------------------------------

            success, frame = camera.read()

            if not success:

                print(
                    "WARNING: camera frame is empty"
                )

                time.sleep(0.05)
                continue

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            person_count = 0

            # ------------------------------------------------
            # Обрабатываем обнаружения
            # ------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    if class_name != "person":
                        continue

                    draw_person_box(
                        frame,
                        box.xyxy[0],
                        confidence,
                    )

                    person_count += 1

            # ------------------------------------------------
            # Текущее состояние
            # ------------------------------------------------

            current_person_present = (
                person_count > 0
            )

            now = time.monotonic()

            # ------------------------------------------------
            # Если YOLO изменил состояние,
            # начинаем отсчёт стабильности.
            # ------------------------------------------------

            if (
                current_person_present
                != candidate_person_present
            ):

                candidate_person_present = (
                    current_person_present
                )

                candidate_since = now

            # ------------------------------------------------
            # Проверяем стабильность.
            # ------------------------------------------------

            state_is_stable = (
                now - candidate_since
                >= STATE_STABLE_SECS
            )

            if (
                state_is_stable
                and candidate_person_present
                != stable_person_present
            ):

                stable_person_present = (
                    candidate_person_present
                )

                if stable_person_present:

                    phrase = "Вижу человека."

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        "STATE: person appeared"
                    )

                else:

                    phrase = "Нет человека."

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        "STATE: person disappeared"
                    )

                # Передаём событие MacSpeaker.
                #
                # Если Mac сейчас говорит или cooldown ещё
                # действует, событие останется pending.
                speaker.submit(phrase)

            # ------------------------------------------------
            # Информация в окне
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"People: {person_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            state_text = (
                "PERSON"
                if stable_person_present
                else "NO PERSON"
            )

            cv2.putText(
                frame,
                f"State: {state_text}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Показываем изображение
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print(
            "AstraMind: interrupted by user."
        )

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print(
            "AstraMind: detector stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()