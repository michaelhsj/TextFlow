from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import mlflow
import torch
import io
import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# --- Pydantic Models for API Data Contracts ---

class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Minimum x-coordinate of the bounding box.")
    y_min: float = Field(..., description="Minimum y-coordinate of the bounding box.")
    x_max: float = Field(..., description="Maximum x-coordinate of the bounding box.")
    y_max: float = Field(..., description="Maximum y-coordinate of the bounding box.")

class OCRPrediction(BaseModel):
    text: str = Field(..., example="HELLO", description="The detected text string.")
    confidence: float = Field(..., example=0.99, description="The model's confidence score for the prediction.")
    box: BoundingBox = Field(..., description="The bounding box surrounding the detected text.")

class OCRResponse(BaseModel):
    filename: str
    predictions: List[OCRPrediction]


# --- App and Model Loading ---

app = FastAPI(
    title="TextFlow API",
    description="API for TextFlow OCR model inference.",
    version="1.0.0"
)

predictor = None

@app.on_event("startup")
def load_model():
    """
    Load the OCR model from MLflow at application startup.
    This uses the MLflow Models URI format to fetch a registered model.
    """
    global predictor
    # Best Practice: Use environment variables to configure the model URI
    # This allows you to switch between "staging" and "production" models easily.
    model_name = os.getenv("MLFLOW_MODEL_NAME", "textflow_db_resnet50")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    model_uri = f"models:/{model_name}/{model_stage}"
    
    print(f"Attempting to load model from: {model_uri}")
    
    try:
        detection_model = mlflow.pytorch.load_model(model_uri)
        # The predictor handles preprocessing, model execution, and postprocessing
        predictor = ocr_predictor(det_arch=detection_model, reco_arch="crnn_vgg16_bn", pretrained=True)
        print("OCR Predictor loaded successfully.")
    except Exception as e:
        predictor = None
        print(f"Failed to load model: {e}")
        print("API will start, but /predict endpoint will be unavailable.")

# --- API Endpoints ---

# Try testing in "[server_url]/docs"

@app.get("/")
def read_root():
    return {"message": "Welcome to the TextFlow API"}

@app.post("/api/v1/predict", response_model=OCRResponse)
async def predict_ocr(file: UploadFile = File(...)):
    """
    Accepts an image file and returns OCR predictions.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not available.")

    # 1. Validate input file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # 2. Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # add preprocessing logic here
    # from your doctr dataset/loader to match training conditions.
    # For now, this is a placeholder.
    # image_tensor = preprocess(image).to(device)
    # The doctr DocumentFile handles in-memory files
    doc = DocumentFile.from_images([image])

    # 3. Run inference using the predictor
    result = predictor(doc)
    
    # 4. Format the results into the Pydantic response model
    predictions = []
    # result.pages[0].blocks is the structure doctr uses
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x_min, y_min), (x_max, y_max) = word.geometry
                predictions.append(OCRPrediction(
                    text=word.value,
                    confidence=word.confidence,
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                ))

    return OCRResponse(filename=file.filename, predictions=predictions)
