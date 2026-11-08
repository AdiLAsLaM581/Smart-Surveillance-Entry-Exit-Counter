# Smart Surveillance: Entry & Exit Counter

A computer vision-based surveillance system that tracks people, counts entry and exit events, and detects unauthorized access to a restricted area.

The project uses YOLOv8 for person detection, ByteTrack for object tracking, and Supervision for line-crossing and zone-based event detection.

## Features

- **Person Detection:** Detects people in surveillance footage using YOLOv8.
- **Object Tracking:** Assigns tracking IDs using ByteTrack.
- **Entry & Exit Counting:** Detects people crossing a virtual line.
- **Restricted-Zone Detection:** Identifies when a tracked person enters a predefined restricted area.
- **Event Logging:** Records detected events in a CSV file.
- **Count Comparison:** Compares automatic counts with manually recorded counts.
- **Video Annotation:** Generates an annotated video showing detections, tracking IDs, and event information.

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- Supervision
- NumPy
- CSV

## How It Works

1. The system reads a surveillance video.
2. YOLOv8 detects people in each frame.
3. ByteTrack assigns and maintains tracking IDs.
4. A virtual line detects entry and exit events.
5. A predefined polygon identifies the restricted area.
6. Detected events and count comparisons are saved for review.

## Project Structure

```text
Smart-Surveillance-Entry-Exit-Counter/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── output/
    ├── events.csv
    └── count_comparison.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AdiLAsLaM581/Smart-Surveillance-Entry-Exit-Counter.git
cd Smart-Surveillance-Entry-Exit-Counter
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the program, configure the following in `main.py`:

- Input video path
- Virtual line coordinates
- Restricted-zone polygon coordinates
- YOLO model path

Place your input video in the project directory or update its path in the code.

## Run the Project

```bash
python main.py
```

## Output Files

| File | Purpose |
|---|---|
| `annotated_video.mp4` | Video with detections, tracking IDs, and event annotations |
| `events.csv` | Log of detected entry, exit, and restricted-zone events |
| `count_comparison.csv` | Comparison between automatic and manual counts |

## Limitations

- Detection and tracking accuracy depend on video quality, camera angle, lighting, and object visibility.
- Incorrect virtual-line or restricted-zone placement can affect event detection.
- Tracking ID switches and missed detections can cause differences between automatic and manual counts.

## Future Improvements

- Real-time CCTV camera integration
- Email or dashboard-based restricted-zone alerts
- Improved tracking accuracy
- Multiple entrance and exit monitoring
- Web-based surveillance dashboard

## Author

**Muhammad Adil**

Information Technology Undergraduate  
Bahria University, Lahore

[GitHub Profile](https://github.com/AdiLAsLaM581)
