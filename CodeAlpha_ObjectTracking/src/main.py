"""
CodeAlpha Internship - Task: Object Detection and Tracking
------------------------------------------------------------
Real-time object detection and multi-object tracking using
Ultralytics YOLOv8 + OpenCV.

Author: Bikki Kumar Rana
------------------------------------------------------------
Run with:
    python src/main.py

Edit the CONFIG section below to change the input source,
model, confidence threshold, or tracker.
"""

import time
from collections import defaultdict

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolov8n.pt"

# Input source:

SOURCE = 0

# Minimum confidence score (0-1) for a detection to be shown.
# Lower = more detections but more false positives.
CONFIDENCE = 0.5

# Image size YOLO resizes frames to before inference. Smaller
# values (e.g. 480) run faster but can miss small objects;
# larger values (e.g. 736) are more accurate but slower.
IMG_SIZE = 640

# Ultralytics ships built-in tracker configs. "bytetrack.yaml"
# is used here - see the explanation section in the README for
# why ByteTrack was chosen over BoT-SORT for this project.
TRACKER = "bytetrack.yaml"


CLASSES_OF_INTEREST = None  # e.g. [0, 2, 3, 5, 7]

# Folder where 'S' key screenshots are saved.
OUTPUT_DIR = "outputs"

# Window title.
WINDOW_NAME = "CodeAlpha - Object Detection & Tracking"


# ============================================================
# HELPER CLASSES
# ============================================================

class FPSTracker:
    """
    Calculates a smoothed, real-time FPS value using a rolling
    average of frame processing times, rather than a single
    hardcoded or one-off measurement.
    """

    def __init__(self, smoothing_window=30):
        self.smoothing_window = smoothing_window
        self.frame_times = []
        self.prev_time = time.time()

    def update(self):
        now = time.time()
        delta = now - self.prev_time
        self.prev_time = now

        if delta > 0:
            self.frame_times.append(1.0 / delta)

        if len(self.frame_times) > self.smoothing_window:
            self.frame_times.pop(0)

        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)


class ObjectCounter:
    """
    Keeps track of:
      - objects currently visible in the frame (per class)
      - the set of unique track IDs seen so far during the run

    IMPORTANT (see README "Known Limitations"): the "seen so
    far" count is based on tracker IDs. If the tracker loses an
    object and re-detects it later, it may assign a new ID,
    which would count the same physical object twice. This is a
    tracker limitation, not something this project claims to
    solve perfectly.
    """

    def __init__(self):
        self.seen_ids_per_class = defaultdict(set)

    def update(self, class_names, track_ids):
        current_counts = defaultdict(int)
        for cls_name, track_id in zip(class_names, track_ids):
            current_counts[cls_name] += 1
            if track_id is not None:
                self.seen_ids_per_class[cls_name].add(track_id)
        return current_counts

    def total_seen(self, cls_name=None):
        if cls_name is not None:
            return len(self.seen_ids_per_class.get(cls_name, set()))
        return sum(len(ids) for ids in self.seen_ids_per_class.values())


# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_model(model_path):
    """Load the YOLO model, raising a clear error on failure."""
    print("[INFO] Loading YOLO model...")
    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"[ERROR] Could not load YOLO model '{model_path}': {exc}")
    print("[INFO] Model loaded successfully.")
    return model


def open_source(source):
    """
    Validate and open the video source (webcam index or file
    path) before handing it to YOLO, so we can give a clean
    error message instead of a cryptic OpenCV failure.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, str):
            raise RuntimeError(f"[ERROR] Could not open video file: {source}")
        raise RuntimeError(f"[ERROR] Could not open webcam (index {source}).")
    cap.release()  # YOLO's own predictor will reopen the source
    return True


def draw_analytics(frame, fps, frame_counts, total_seen, num_classes):
    """Draw a clean analytics panel in the top-left corner."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Tracked Objects: {sum(frame_counts.values())}",
        f"Persons: {frame_counts.get('person', 0)}",
        f"Cars: {frame_counts.get('car', 0)}",
        f"Classes in frame: {num_classes}",
        f"Total unique IDs seen: {total_seen}",
    ]

    panel_width = 260
    panel_height = 20 + 22 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 28
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22

    return frame


def draw_detections(frame, boxes, class_names_map, track_ids, confidences, class_ids):
    """Draw bounding boxes with class name, tracking ID and confidence."""
    class_names_out = []

    for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_name = class_names_map.get(int(cls_id), str(cls_id))
        class_names_out.append(cls_name)

        color = (255, 128, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        id_text = f"ID:{track_id}" if track_id is not None else "ID:--"
        label = f"{cls_name} {id_text} {conf:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, class_names_out


def run():
    try:
        open_source(SOURCE)
    except RuntimeError as exc:
        print(exc)
        return

    try:
        model = load_model(MODEL_PATH)
    except RuntimeError as exc:
        print(exc)
        return

    class_names_map = model.names  # dict: {0: 'person', 2: 'car', ...}
    fps_tracker = FPSTracker()
    counter = ObjectCounter()
    paused = False

    print("[INFO] Starting object detection and tracking...")
    print("[INFO] Controls -> Q: Quit | S: Save frame | P: Pause/Resume")

    try:
        results_generator = model.track(
            source=SOURCE,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            tracker=TRACKER,
            classes=CLASSES_OF_INTEREST,
            stream=True,
            persist=True,
            verbose=False,
        )

        for result in results_generator:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                continue

            frame = result.orig_img.copy()

            boxes_obj = result.boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                boxes_xyxy = boxes_obj.xyxy.cpu().numpy()
                confidences = boxes_obj.conf.cpu().numpy()
                class_ids = boxes_obj.cls.cpu().numpy()

                if boxes_obj.id is not None:
                    track_ids = boxes_obj.id.int().cpu().tolist()
                else:
                    # Tracker did not assign IDs this frame
                    track_ids = [None] * len(boxes_xyxy)
            else:
                boxes_xyxy, confidences, class_ids, track_ids = [], [], [], []

            frame, class_names_out = draw_detections(
                frame, boxes_xyxy, class_names_map, track_ids, confidences, class_ids
            )

            frame_counts = counter.update(class_names_out, track_ids)
            fps = fps_tracker.update()
            total_seen = counter.total_seen()
            num_classes = len(frame_counts)

            frame = draw_analytics(frame, fps, frame_counts, total_seen, num_classes)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit key pressed. Exiting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{OUTPUT_DIR}/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Frame saved to {filename}")
            elif key == ord('p'):
                paused = True
                print("[INFO] Paused. Press 'P' again to resume.")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as exc:
        print(f"[ERROR] An unexpected error occurred: {exc}")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Program ended.")


if __name__ == "__main__":
    run()
