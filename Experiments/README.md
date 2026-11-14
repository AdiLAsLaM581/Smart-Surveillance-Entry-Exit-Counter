# Week 6 – Model Evaluation and Experimentation

## 1. Objective

The objective of Week 6 was to evaluate the existing Smart Surveillance Entry/Exit Counter system and improve its performance through controlled experiments. The baseline configuration was compared with three alternative configurations by changing one setting at a time.

## 2. Testing Setup

- **Input video:** Same surveillance video used in the Week 5 implementation
- **Frames processed:** 360
- **Manual entry count:** 3
- **Manual exit count:** 5
- **Total manual crossings:** 8
- **Baseline model:** YOLOv8n (`yolov8n.pt`)
- **Tracking algorithm:** ByteTrack

## 3. Experiments and Results

| Experiment | Model | Confidence | Image Size | Automatic Entries | Automatic Exits | Count Error | Processing FPS |
|---|---|---|---|---:|---:|---:|---:|
| Baseline | YOLOv8n | Default | Default | 2 | 1 | 62.5% | 4.36 |
| Confidence 0.4 | YOLOv8n | 0.4 | Default | 3 | 1 | 50.0% | 4.76 |
| Image Size 960 | YOLOv8n | Default | 960 | 3 | 2 | 37.5% | 2.43 |
| YOLO11s Model | YOLO11s | Default | Default | 2 | 1 | 62.5% | 1.58 |

## 4. Count Error Calculation

Count error was calculated using:

`Count Error (%) = (|Automatic Total − Manual Total| / Manual Total) × 100`

The manual total was 8 crossings (3 entries + 5 exits).

**Examples:**

- Baseline: |3 − 8| / 8 × 100 = 62.5%
- Confidence 0.4: |4 − 8| / 8 × 100 = 50%
- Image Size 960: |5 − 8| / 8 × 100 = 37.5%
- YOLO11s: |3 − 8| / 8 × 100 = 62.5%

## 5. Observations

### Baseline

The original YOLOv8n configuration processed the video at 4.36 FPS. It detected 2 entries and 1 exit, resulting in a 62.5% count error.

### Confidence 0.4

Increasing the confidence threshold to 0.4 improved the entry count from 2 to 3 and reduced the count error to 50%. It also achieved the highest processing speed at 4.76 FPS.

### Image Size 960

Increasing the image size to 960 improved the automatic counts to 3 entries and 2 exits. This configuration achieved the lowest count error of 37.5%. However, processing speed decreased to 2.43 FPS.

### YOLO11s Model

Changing the model from YOLOv8n to YOLO11s did not improve the automatic counts. It produced a 62.5% count error and the lowest processing speed of 1.58 FPS.

## 6. Conclusion

The experiments showed that changing model settings affects both counting accuracy and processing speed.

The Image Size 960 configuration achieved the best counting result among the tested configurations, reducing the count error from 62.5% to 37.5%. However, it required considerably more processing time.

The Confidence 0.4 configuration provided a useful balance, improving the count error to 50% while achieving the highest processing speed.

YOLO11s did not provide an improvement for this particular video and hardware setup.

Overall, the experiments demonstrate that model configuration should be selected according to the required balance between counting accuracy and processing speed.

## 7. Limitations

The system still undercounted exits in all four experiments. Possible reasons include missed detections, tracking ID switches, people crossing together, and the position of the virtual counting line. Further testing with different tracker settings and line placements may help improve the results.

*Note: These results are based on one 360-frame test video and should not be treated as a general performance guarantee.*
