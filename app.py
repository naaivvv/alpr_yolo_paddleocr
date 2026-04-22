import base64
import cv2
import logging
import numpy as np
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.pipeline import ALPRPipeline
from src.utils import draw_results

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="ALPR Vision System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Pipeline
MODEL_PATH = "models/yolo26_custom.pt"
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = ALPRPipeline(yolo_model_path=MODEL_PATH)
        logging.info("ALPR Pipeline initialized successfully.")
    else:
        logging.error(f"Model not found at {MODEL_PATH}. Pipeline initialization failed.")

class InferenceResult(BaseModel):
    image_base64: str
    results: list

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("templates/index.html")

@app.post("/inference")
async def run_inference(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="ALPR Pipeline not initialized.")
    
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Save temporary file for pipeline (since it expects a path in some methods, 
        # though process_image in pipeline.py uses cv2.imread(image_path))
        # Let's modify pipeline to accept image array if needed, but for now 
        # we'll save a temp file to be safe and compatible with current pipeline.
        temp_path = "output/temp_upload.jpg"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(temp_path, img)
        
        # Process image
        original_img, results = pipeline.process_image(temp_path)
        
        if original_img is None:
            raise HTTPException(status_code=500, detail="Inference failed.")

        # Draw results
        annotated_img = draw_results(original_img, results)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image_base64": f"data:image/jpeg;base64,{img_base64}",
            "results": results
        }
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
