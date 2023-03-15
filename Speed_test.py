import cv2
import numpy as np
from deepsort import DeepSort
from PIL import Image

# Initialize DeepSort object
deepsort = DeepSort('model_path')

# Initialize variables
frame_num = 0
prev_frame_data = {}
output_data = {}

# Read video
cap = cv2.VideoCapture('video_path')

# Loop through frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply geometry normalization function
    # (replace with your own normalization function)
    pil_image = geometry_normalization(pil_image)

    # Convert PIL Image to numpy array
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Detect objects and update tracker
    bbox_xywh, confidences, clss = yolo_detection(frame)
    outputs = deepsort.update(bbox_xywh, confidences, clss, frame)

    # Calculate and output speed for each ID
    for output in outputs:
        x1, y1, x2, y2, cls, track_id = output

        # Check if ID is new or disappeared
        if track_id not in prev_frame_data:
            # New ID, save position and frame number
            prev_frame_data[track_id] = (x1, y1, frame_num)
        else:
            # Calculate speed and save output data
            prev_x, prev_y, prev_frame = prev_frame_data[track_id]
            distance = np.sqrt((x1-prev_x)**2 + (y1-prev_y)**2)
            time_diff = frame_num - prev_frame
            speed = distance / time_diff
            output_data[track_id] = speed
            print(f"ID {track_id} speed: {speed}")

            # Update previous frame data with current position and frame number
            prev_frame_data[track_id] = (x1, y1, frame_num)

    # Print output data for each ID when ID disappears
    for track_id in list(prev_frame_data.keys()):
        if track_id not in output_data:
            print(f"ID {track_id} speed: N/A (disappeared)")

# Release video and exit
cap.release()
cv2.destroyAllWindows()

