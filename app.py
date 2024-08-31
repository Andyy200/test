from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO  # Assuming YOLOv10 is supported here
import os
app = Flask(__name__)

# Initialize the YOLOv10 model (replace 'yolov10n.pt' with the actual model path)
model = YOLO('yolov10n.pt')  # Replace with the actual path to the YOLOv10 model

cap = None  # Initialize the cap variable


def generate_frames():
    global cap
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


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', connected=False)


@app.route('/connect', methods=['POST'])
def connect():
    global cap
    rtsp_url = request.form['rtsp_url']
    
    # Release any previous capture if it exists
    if cap is not None:
        cap.release()
    
    # Initialize the video capture with the new RTSP URL
    cap = cv2.VideoCapture(rtsp_url)
    
    # Check if the connection was successful
    if not cap.isOpened():
        return "Error: Unable to open video stream. Please check the URL and try again."
    
    return render_template('index.html', connected=True)


@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        # Release the video capture when the application stops
        if cap is not None:
            cap.release()
