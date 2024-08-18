from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from yolo_predictions import YOLO_Pred
from fastapi.responses import StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your Flutter app's domain/IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the YOLO model
yolo = YOLO_Pred('./Model2/weights/best.onnx', 'data.yaml')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO predictions
    pred_image = yolo.predictions(frame)

    # Convert the result to a format suitable for response
    _, img_encoded = cv2.imencode('.jpg', pred_image)
    img_bytes = img_encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

