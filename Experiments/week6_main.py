import cv2
import numpy as np
import supervision as sv
import csv
import os
import time
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = r"E:\Zee Project\EntryPoint\Surveillance_test_video_for_pede…_5cfc95bd.mp4"

# Select ONE experiment and run the program.
# Run the program four times, changing only EXPERIMENT each time:
# Baseline, Confidence_0.4, Image_Size_960, YOLO11s_Model
EXPERIMENT = "YOLO11s_Model"

# Week 5 baseline settings
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = None
IMAGE_SIZE = None

# Week 6 experiment settings
if EXPERIMENT == "Baseline":
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE = None
    IMAGE_SIZE = None

elif EXPERIMENT == "Confidence_0.4":
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE = 0.4
    IMAGE_SIZE = None

elif EXPERIMENT == "Image_Size_960":
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE = None
    IMAGE_SIZE = 960

elif EXPERIMENT == "YOLO11s_Model":
    MODEL_PATH = "yolo11s.pt"
    CONFIDENCE = None
    IMAGE_SIZE = None

else:
    raise ValueError("Invalid EXPERIMENT name.")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR, EXPERIMENT + "_annotated_video.mp4"
)
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, EXPERIMENT + "_events.csv"
)
COMPARISON_CSV = os.path.join(OUTPUT_DIR, "week6_comparison.csv")

# ---------------- LOAD MODEL AND VIDEO ----------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError("Unable to open input video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
ret, frame = cap.read()

if not ret:
    raise ValueError("Unable to read video.")

height, width = frame.shape[:2]

# ---------------- DEFINE LINE AND RESTRICTED ZONE ----------------

# Horizontal virtual line for entry/exit counting.
line_start = sv.Point(0, int(height * 0.50))
line_end = sv.Point(width, int(height * 0.50))

line_zone = sv.LineZone(
    start=line_start,
    end=line_end
)

# Polygon defining the restricted area.
polygon = np.array([
    [int(width * 0.65), int(height * 0.20)],
    [int(width * 0.95), int(height * 0.20)],
    [int(width * 0.95), int(height * 0.85)],
    [int(width * 0.65), int(height * 0.85)]
])

restricted_zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# ---------------- OUTPUT VIDEO ----------------

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Unable to create output video.")

# ---------------- EVENT VARIABLES ----------------

events = []
previous_inside = set()
frame_number = 0

print("Processing video...")
print("Experiment:", EXPERIMENT)
print("Model:", MODEL_PATH)
print("Confidence:", CONFIDENCE if CONFIDENCE is not None else "Default")
print("Image size:", IMAGE_SIZE if IMAGE_SIZE is not None else "Default")

# Start measuring processing speed.
start_time = time.perf_counter()

# ---------------- MAIN PROCESSING LOOP ----------------

while ret:

    frame_number += 1

    # Detect people and maintain their tracking IDs.
    # Week 5 tracking settings are preserved.
    track_settings = {
        "persist": True,
        "classes": [0],
        "tracker": "bytetrack.yaml",
        "verbose": False
    }

    # Apply only the setting being tested in this experiment.
    if CONFIDENCE is not None:
        track_settings["conf"] = CONFIDENCE

    if IMAGE_SIZE is not None:
        track_settings["imgsz"] = IMAGE_SIZE

    result = model.track(frame, **track_settings)[0]

    detections = sv.Detections.from_ultralytics(result)

    # Keep only person detections.
    detections = detections[detections.class_id == 0]

    # Default zone mask for detected people.
    zone_mask = np.zeros(len(detections), dtype=bool)

    # Process events only when tracking IDs are available.
    if len(detections) and detections.tracker_id is not None:

        # Check line crossings and update entry/exit counts.
        crossed_in, crossed_out = line_zone.trigger(detections)

        for track_id, entered, exited in zip(
            detections.tracker_id, crossed_in, crossed_out
        ):
            if entered:
                events.append([
                    frame_number, round(frame_number / fps, 2),
                    "ENTRY", int(track_id), datetime.now().isoformat()
                ])
                print(f"ENTRY: Person {track_id}")

            if exited:
                events.append([
                    frame_number, round(frame_number / fps, 2),
                    "EXIT", int(track_id), datetime.now().isoformat()
                ])
                print(f"EXIT: Person {track_id}")

        # Detect people inside the restricted polygon.
        zone_mask = restricted_zone.trigger(detections)

        current_inside = {
            int(track_id)
            for track_id, inside in zip(detections.tracker_id, zone_mask)
            if inside
        }

        # Generate alerts only for newly entered people.
        for track_id in current_inside - previous_inside:

            events.append([
                frame_number, round(frame_number / fps, 2),
                "RESTRICTED_ZONE_ENTRY", track_id,
                datetime.now().isoformat()
            ])

            print(f"RESTRICTED ZONE ALERT: Person {track_id}")

        previous_inside = current_inside

    else:
        previous_inside = set()

    # ---------------- DRAW ANNOTATIONS ----------------

    # Draw transparent restricted area and its boundary.
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (0, 0, 255))
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

    cv2.polylines(frame, [polygon], True, (0, 0, 255), 3)
    cv2.line(
        frame,
        (line_start.x, line_start.y),
        (line_end.x, line_end.y),
        (255, 255, 0), 3
    )

    # Draw person bounding boxes and tracking IDs.
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):

        track_id = (
            int(detections.tracker_id[i])
            if detections.tracker_id is not None else "?"
        )

        color = (0, 0, 255) if zone_mask[i] else (0, 255, 0)

        cv2.rectangle(
            frame, (int(x1), int(y1)),
            (int(x2), int(y2)), color, 2
        )

        cv2.putText(
            frame, f"ID: {track_id}",
            (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # Display event counts on the video.
    cv2.putText(
        frame, f"ENTRY: {line_zone.in_count}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"EXIT: {line_zone.out_count}",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    cv2.putText(
        frame, f"INSIDE ZONE: {len(previous_inside)}",
        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    # Display the experiment name.
    cv2.putText(
        frame, f"EXPERIMENT: {EXPERIMENT}",
        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Save and display processed frame.
    out.write(frame)
    cv2.imshow("Week 6 - " + EXPERIMENT, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    ret, frame = cap.read()

# ---------------- SAVE RESULTS ----------------

elapsed_time = time.perf_counter() - start_time

cap.release()
out.release()
cv2.destroyAllWindows()

# Save all detected events into a CSV file.
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)
    writer.writerow([
        "Frame", "Time (Seconds)", "Event",
        "Tracking ID", "Timestamp"
    ])
    writer.writerows(events)

# ---------------- MANUAL COUNT COMPARISON ----------------

automatic_entry = line_zone.in_count
automatic_exit = line_zone.out_count

print("\n========== AUTOMATIC RESULTS ==========")
print("Entries:", automatic_entry)
print("Exits:", automatic_exit)
print("Restricted zone alerts:",
      sum(e[2] == "RESTRICTED_ZONE_ENTRY" for e in events))

# Calculate processing FPS.
processing_fps = frame_number / elapsed_time if elapsed_time > 0 else 0

print("Frames processed:", frame_number)
print("Processing time:", round(elapsed_time, 2), "seconds")
print("Processing FPS:", round(processing_fps, 2))

try:
    manual_entry = int(input("\nEnter manual entry count: "))
    manual_exit = int(input("Enter manual exit count: "))

    print("\n========== COMPARISON ==========")
    print(f"Entries: Automatic={automatic_entry}, Manual={manual_entry}, Difference={abs(automatic_entry-manual_entry)}")
    print(f"Exits: Automatic={automatic_exit}, Manual={manual_exit}, Difference={abs(automatic_exit-manual_exit)}")

    if automatic_entry == manual_entry and automatic_exit == manual_exit:
        print("Both counts match!")
    else:
        print("Differences may be caused by ID switches, groups crossing together, missed detections or incorrect line placement.")

    # Week 6 count error: compare total crossings with manual total.
    manual_total = manual_entry + manual_exit
    automatic_total = automatic_entry + automatic_exit

    if manual_total > 0:
        count_error = (
            abs(automatic_total - manual_total) / manual_total
        ) * 100
    else:
        count_error = 0

    print("Count error:", round(count_error, 2), "%")

    # Save automatic and manual counts in the original comparison CSV.
    with open(
        os.path.join(OUTPUT_DIR, EXPERIMENT + "_count_comparison.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Event",
            "Automatic Count",
            "Manual Count",
            "Difference"
        ])

        writer.writerow([
            "Entries",
            automatic_entry,
            manual_entry,
            abs(automatic_entry - manual_entry)
        ])

        writer.writerow([
            "Exits",
            automatic_exit,
            manual_exit,
            abs(automatic_exit - manual_exit)
        ])

    # Add this experiment to the Week 6 before-and-after table.
    comparison_row = [
        EXPERIMENT,
        MODEL_PATH,
        CONFIDENCE if CONFIDENCE is not None else "Default",
        IMAGE_SIZE if IMAGE_SIZE is not None else "Default",
        automatic_entry,
        manual_entry,
        automatic_exit,
        manual_exit,
        round(count_error, 2),
        round(processing_fps, 2)
    ]

    file_exists = os.path.isfile(COMPARISON_CSV)

    with open(COMPARISON_CSV, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Version", "Model", "Confidence", "Image Size",
                "Automatic Entry", "Manual Entry",
                "Automatic Exit", "Manual Exit",
                "Count Error (%)", "FPS"
            ])

        # Avoid adding the same experiment twice.
        writer.writerow(comparison_row)

except ValueError:
    print("Invalid manual count. Comparison table was not updated.")

print("\nProject completed!")
print("Video:", OUTPUT_VIDEO)
print("Events:", OUTPUT_CSV)
print("Week 6 comparison:", COMPARISON_CSV)
