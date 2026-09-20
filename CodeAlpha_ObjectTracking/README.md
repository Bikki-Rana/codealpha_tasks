# Object Detection and Tracking using YOLO and OpenCV

## Description

This project was built for the **CodeAlpha Artificial Intelligence Internship**, for the task *"Object Detection and Tracking."* It's a real-time system that detects objects from a webcam feed or a video file, tracks each object across frames, and assigns it a persistent ID so it can be followed as it moves around the scene.

Under the hood, it uses **YOLOv8** (via the Ultralytics library) for detection and **ByteTrack** for multi-object tracking, wrapped in a simple OpenCV display loop with on-screen analytics.

## Features

- Real-time object detection on webcam or video file input
- Multi-object tracking with persistent IDs across frames
- Confidence score shown for every detection
- Live FPS monitoring (calculated from actual frame times, not hardcoded)
- On-screen analytics: total tracked objects, person count, car count, number of classes present
- Keyboard controls to quit, save a screenshot, or pause/resume
- Clean, modular, commented code — no giant single function

## Technologies

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack (via Ultralytics' built-in tracker configs)
- NumPy

## Installation

1. Clone or download this repository, then open the folder in VS Code.

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate it.

   Windows:
   ```bash
   venv\Scripts\activate
   ```
   macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

From the project root, with the virtual environment activated:

```bash
python src/main.py
```

The first run will automatically download `yolov8n.pt` (the small/nano YOLOv8 model, ~6 MB) through Ultralytics — no manual download needed.

To use a video file instead of the webcam, open `src/main.py` and change:
```python
SOURCE = 0
```
to
```python
SOURCE = "videos/input.mp4"
```

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the program |
| `S` | Save the current frame to `outputs/` |
| `P` | Pause / resume playback |

## Project Structure

```text
CodeAlpha_ObjectTracking/
│
├── src/
│   └── main.py
│
├── models/
│
├── videos/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## How It Works

```text
Video/Webcam
      ↓
YOLO Detection (yolov8n.pt)
      ↓
Object Bounding Boxes + Class + Confidence
      ↓
ByteTrack Tracker
      ↓
Persistent Tracking IDs
      ↓
Analytics (FPS, counts, classes)
      ↓
Display / Output (OpenCV window)
```

Each frame from the source is passed through the YOLOv8 model, which outputs bounding boxes, class labels, and confidence scores. Ultralytics' built-in tracker (ByteTrack) then matches these detections to existing tracks from previous frames using motion and appearance cues, assigning a stable ID to each object. The results are drawn on the frame along with a live analytics panel, and the frame is shown in an OpenCV window.

## Known Limitations

Being transparent about what this project does and doesn't do:

- Tracking IDs are preserved reasonably well while an object stays visible, but if an object leaves the frame and re-enters, or gets fully occluded for a long time, ByteTrack may assign it a **new** ID. This is a known limitation of appearance-free trackers like ByteTrack, not something this project claims to solve.
- The "total unique IDs seen" analytic counts *tracker IDs*, not necessarily unique physical objects — for the reason above, it can overcount if the same object re-enters the frame.
- Detection accuracy depends on lighting, camera angle, and object size — `yolov8n.pt` is the smallest YOLOv8 model, chosen for real-time speed on a normal laptop CPU, at some cost to accuracy versus larger variants (`yolov8s/m/l/x.pt`).

## Future Improvements

- Line-crossing detection (e.g., counting people crossing a doorway)
- Dedicated vehicle counting
- Speed estimation for moving objects
- Zone-based analytics (define regions of interest)
- Dedicated people-counting mode
- Heatmaps of object movement over time
- Recording annotated output video to disk
- A simple web dashboard for analytics instead of the OpenCV window

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
