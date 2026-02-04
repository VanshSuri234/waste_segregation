import logging
import torch
import functools
import json
import shutil
import os
from pathlib import Path
from typing import List  # Required for multi-file upload
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # 1. Import this
load_dotenv()
# --- FIX: Global Security Override ---
torch.load = functools.partial(torch.load, weights_only=False)

from services.yolo_parser import YOLOParser
from services.llm_service import LLMService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

yolo_parser = None
llm_service = None

# Folder Setup
UPLOAD_DIR = Path("inputs")
PROCESSED_DIR = Path("processed_data")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_parser, llm_service
    logger.info("🚀 STARTING NTPC WASTE ANALYSIS SYSTEM")
    try:
        local_model = Path("model/best.pt")
        model_to_load = str(local_model) if local_model.exists() else "yolov8n-seg.pt"
        yolo_parser = YOLOParser(model_to_load)
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            logger.warning("⚠️ No API Key found in environment variables!")
            # Optional: Fallback for local testing only
            # api_key = "your_key_here"
        # Initialize LLM with your Groq Key
        llm_service = LLMService(api_key=api_key)
        logger.info("✓ System Ready. Waiting for Frontend Uploads...")
        yield 
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        raise

app = FastAPI(title="Waste Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "yolo": "ready", "llm": "ready"}

@app.post("/upload")
async def upload_waste(file: UploadFile = File(...)): # Changed from List to single file
    try:
        # 1. Save Original
        input_path = UPLOAD_DIR / file.filename
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Process with YOLO
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            data = yolo_parser.process_video(input_path)
        else:
            data = yolo_parser.process_image(input_path)
        
        # 3. Create a list of one for the LLM (so your LLM service logic stays the same)
        batch_results = [data]

        industrial_context = {
            "avg_cycle_time": "2.4s",
            "robot_accuracy": "94%",
            "environmental_temp": "32°C",
            "thermal_sensor_status": "Normal"
        }

        # 4. Generate Report
        summary_report = llm_service.generate_segregation_report(batch_results, industrial_context)

        return {
            "status": "success",
            "file_processed": file.filename,
            "report": summary_report
        }

    except Exception as e:
        logger.error(f"Batch Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_query(query: str = Form(...)):
    # This uses the RAW JSON data stored in llm_service memory
    response = llm_service.ask_general_question(query) 
    return {"analysis": response}

if __name__ == "__main__":
    import uvicorn
    import os
    # Get port from environment variable, default to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)