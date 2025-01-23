from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from yolo_predictions import YOLO_Pred
from starlette.responses import Response, StreamingResponse
import io
from PIL import Image
import numpy as np
import json
import cv2
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import base64

# Initialize and obtain the model
yolo = YOLO_Pred('./Model2/weights/best.onnx', 'data.yaml')

# FastAPI application setup
app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

# CORS (Cross-Origin Resource Sharing) middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/notify/v1/health')
def get_health():
    return dict(msg='OK')


# Convert bytes to a NumPy array
def bytes_to_numpy(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data))
    return np.array(image)


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    # Convert bytes to NumPy array
    input_image = bytes_to_numpy(file)
    results = yolo.predictions(input_image)

    # Convert NumPy array to a list
    detect_res = results.tolist()

    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    # Convert bytes to NumPy array
    input_image = bytes_to_numpy(file)
    #only processed image
    # results = yolo.predictions(input_image)
    #image with data
    processed_image, detect_res = yolo.predictions(input_image)

    # Save the result image
    bytes_io = io.BytesIO()
    img_base64 = Image.fromarray(processed_image)
    img_base64.save(bytes_io, format="jpeg")
    img_str = base64.b64encode(bytes_io.getvalue()).decode('utf-8')

    return {
        "image": img_str,
        "detections": detect_res
    }

@app.post("/object-to-video")
async def detect_objects_from_video(file: UploadFile = File(...)):
    # Save the uploaded video file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with temp_file as f:
            f.write(await file.read())

        # Open the video file with OpenCV
        video = cv2.VideoCapture(temp_file.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Prepare output video stream
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # YOLO prediction on each frame
            results = yolo.predictions(frame)

            # Save the result frame
            out.write(results)

        video.release()
        out.release()

        # Return the processed video
        return StreamingResponse(io.BytesIO(open('output.mp4', 'rb').read()), media_type="video/mp4")

    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)
        if os.path.exists('output.mp4'):
            os.remove('output.mp4')

@app.websocket("/ws/realtime-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive frame data from the client
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data))
            frame = np.array(image)

            # Process the frame with YOLO
            processed_frame, detections = yolo.predictions(frame)
            distances = yolo.compute_manhattan_distance(detections)

            # Convert results to an image
            result_image = Image.fromarray(processed_frame)
            buffer = io.BytesIO()
            result_image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

            # Correct indentation for response_payload
            response_payload = {
                "detections": detections,
                "distances": distances
            }

            # Send processed frame back to the client
            await websocket.send_bytes(img_bytes)
            await websocket.send_text(json.dumps(response_payload))

        except WebSocketDisconnect:
            print("Client disconnected")
            break
