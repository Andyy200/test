from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO  # Assuming YOLOv10 is supported here

app = Flask(__name__)

# Initialize the YOLOv10 model (replace 'yolov10n.pt' with the actual model path)
model = YOLO('yolov10n.pt')  # Replace with the actual path to the YOLOv10 model

# Video capture setup (0 for webcam)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the YOLO model to detect objects in the frame
        results = model(frame)
        
        # Draw bounding boxes on the detected objects
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = conf.item()
                class_id = int(cls.item())
                class_name = model.names[class_id]
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame as a JPEG image and yield it as a byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()

# Release the video capture when the application stops
cap.release()
