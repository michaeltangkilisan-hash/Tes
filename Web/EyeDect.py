import cv2
import mediapipe as mp
import math
import time
import threading
from playsound import playsound
import numpy as np

def play_alarm():
    threading.Thread(target=playsound, args=("alarm.mp3",), daemon=True).start()

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def eye_aspect_ratio(landmarks, idxs):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idxs]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0
    return (vertical1 + vertical2) / (2.0 * horizontal)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22
SLEEP_TIME_THRESHOLD = 3.0
start_closed_time = None
alarm_played = False

def run_detection(mode: str = "1", mata: str = "2"):
    """Jalankan sistem pendeteksi mata.

    mode:
        "1" -> Berwarna (RGB)
        "2" -> Keabuan (Grayscale Manual)
    mata:
        "1" -> Deteksi 1 mata (kiri ATAU kanan)
        "2" -> Deteksi 2 mata sekaligus
    """
    global start_closed_time, alarm_played

    if mode == "1":
        mode_text = "BERWARNA"
    elif mode == "2":
        mode_text = "KEABUAN MANUAL"
    else:
        print("Pilihan mode tidak valid. Default: BERWARNA")
        mode = "1"
        mode_text = "BERWARNA"

    if mata not in ("1", "2"):
        print("Pilihan jumlah mata tidak valid. Default: 2 mata")
        mata = "2"

    brightness = 1.0

    def nothing(x):
        pass

    if mode == "2":
        cv2.namedWindow("Sistem Pendeteksi Mata")
        cv2.createTrackbar("Brightness", "Sistem Pendeteksi Mata", 100, 200, nothing)
        print("\nGunakan trackbar untuk atur tingkat kecerahan (50–200)\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak terdeteksi!")

    print(f"\nMode aktif: {mode_text}")
    print("Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        frame = cv2.resize(frame, (640, 480))

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

        if mode == "2":
            brightness_value = cv2.getTrackbarPos("Brightness", "Sistem Pendeteksi Mata")
            brightness = brightness_value / 100.0

            b, g, r = cv2.split(frame_blur.astype(np.float32))
            gray_manual = 0.299 * r + 0.587 * g + 0.114 * b
            gray_manual = cv2.convertScaleAbs(gray_manual, alpha=brightness)

            gray_manual_eq = cv2.equalizeHist(gray_manual)

            kernel = np.ones((3, 3), np.uint8)
            morph_gradient = cv2.morphologyEx(gray_manual_eq, cv2.MORPH_GRADIENT, kernel)

            sobelx = cv2.Sobel(gray_manual, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_manual, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = cv2.magnitude(sobelx, sobely)
            sobel_mag = cv2.convertScaleAbs(sobel_mag)

            canny = cv2.Canny(gray_manual, 80, 150)

            combined_edges = cv2.addWeighted(morph_gradient, 0.5, sobel_mag, 0.5, 0)
            combined_edges = cv2.addWeighted(combined_edges, 0.7, canny, 0.3, 0)

            display_frame = cv2.cvtColor(gray_manual, cv2.COLOR_GRAY2BGR)
            display_frame[:, :, 2] = cv2.max(display_frame[:, :, 2], combined_edges)

            rgb_frame = display_frame.copy()

        else:
            rgb_frame = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)
            display_frame = frame.copy()

        display_frame = cv2.flip(display_frame, 1)
        rgb_frame = cv2.flip(rgb_frame, 1)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)

            # Pilih cara perhitungan EAR berdasarkan jumlah mata yang dideteksi
            if mata == "1":
                # Gunakan mata dengan EAR lebih besar (diasumsikan mata yang aktif terlihat jelas)
                if left_ear >= right_ear:
                    ear = left_ear
                    eye_label = "Mata Kiri"
                    eye_indices_to_draw = LEFT_EYE_IDX
                else:
                    ear = right_ear
                    eye_label = "Mata Kanan"
                    eye_indices_to_draw = RIGHT_EYE_IDX
            else:
                ear = (left_ear + right_ear) / 2.0
                eye_label = "2 Mata"
                eye_indices_to_draw = LEFT_EYE_IDX + RIGHT_EYE_IDX

            for idx in eye_indices_to_draw:
                cv2.circle(display_frame, landmarks[idx], 2, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                if start_closed_time is None:
                    start_closed_time = time.time()
                elapsed = time.time() - start_closed_time

                if elapsed >= SLEEP_TIME_THRESHOLD and not alarm_played:
                    cv2.putText(
                        display_frame,
                        "!!! MATA TERTUTUP TERDETEKSI !!!",
                        (50, 80),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    play_alarm()
                    alarm_played = True
            else:
                start_closed_time = None
                alarm_played = False

            cv2.putText(
                display_frame,
                f"EAR ({eye_label}): {ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255),
                2,
            )
        else:
            start_closed_time = None
            alarm_played = False

        deteksi_text = "1 Mata (kiri/kanan)" if mata == "1" else "2 Mata"
        cv2.putText(
            display_frame,
            f"Mode: {mode_text} | Deteksi: {deteksi_text}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Sistem Pendeteksi Mata", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== Sistem Pendeteksi Mata (berbasis pengolahan citra digital) ===")
    print("Pilih mode tampilan:")
    print("1. Berwarna (RGB)")
    print("2. Keabuan (Grayscale Manual)")

    mode = input("Masukkan pilihan (1/2): ")

    if mode not in ("1", "2"):
        print("Pilihan tidak valid. Default: BERWARNA")
        mode = "1"

    run_detection(mode)
