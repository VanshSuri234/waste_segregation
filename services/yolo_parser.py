import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Directory setup
ROOT_DIR = Path(__file__).resolve().parent.parent 
PROCESSED_DIR = ROOT_DIR / "processed_data"

class YOLOParser:
    def __init__(self, model_path):
        # 1. Load the model
        print(f"🚀 Initializing YOLO Engine with: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        # 2. Define NTPC Specific Class Mapping
        self.ntpc_class_map = {
            0: "Cardboard",
            1: "Large Plastic Container/Pipe",
            2: "Styrofoam Plate/Tray",
            3: "White Plastic Packaging",
            4: "Colored Packaging (Red/Pink)",
            5: "Small Plastic Debris",
            6: "White Plastic Plate/Lid",
            7: "Green Plastic Bottle",
            8: "Straw/Stick/Tube",
            9: "Textile/Fabric",
            10: "Plastic Bag (White/Clear)",
            11: "Mixed Plastic Waste",
            12: "Plastic Bottle with Label",
            13: "General Plastic Waste",
            14: "Large Waste Pile/Area"
        }
        
        # FIX: Do NOT try to set self.model.names directly. 
        # Instead, we use our local map for all downstream logic.
        self.class_names = self.ntpc_class_map
        
        logger.info(f"YOLO Engine Initialized with {len(self.class_names)} custom classes")

    def process_image(self, img_path):
        """Processes an image and returns high-detail data for LLM analysis"""
        # Note: We pass 'names' in the predict call if we want custom labels on saved images
        results = self.model.predict(
            source=str(img_path), 
            conf=0.25, 
            save=True, 
            project=str(PROCESSED_DIR), 
            name="detections", 
            exist_ok=True,
            verbose=False
        )
        
        detections_data = []
        result = results[0]
        img_h, img_w = result.orig_shape

        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0])
                # We use our custom map here
                name = self.class_names.get(class_id, f"Unknown({class_id})")
                
                coords = box.xyxy[0].tolist() 
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                
                center_x = coords[0] + (width / 2)
                center_y = coords[1] + (height / 2)
                
                detections_data.append({
                    "class_name": name,
                    "class_id": class_id,
                    "confidence": round(float(box.conf[0]), 4),
                    "area_px": round(width * height, 2),
                    "dimensions": {"w": round(width, 2), "h": round(height, 2)},
                    "center": [round(center_x, 2), round(center_y, 2)],
                    "aspect_ratio": round(width / height, 2) if height != 0 else 0
                })

        return {
            "file_name": str(img_path.name),
            "image_size": [img_w, img_h],
            "total_count": len(detections_data),
            "detections": detections_data,
            "type": "image"
        }

    def process_video(self, video_path):
        results = self.model.predict(
            source=str(video_path),
            conf=0.25,
            save=True,
            project=str(PROCESSED_DIR),
            name="video_detections",
            exist_ok=True,
            stream=True 
        )
        
        detections_data = []
        frame_idx = 0
        for result in results:
            frame_idx += 1
            if frame_idx % 30 == 0 and result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    name = self.class_names.get(class_id, f"Unknown({class_id})")
                    detections_data.append({
                        "class_name": name,
                        "class_id": class_id,
                        "confidence": round(float(box.conf[0]), 4),
                        "frame": frame_idx
                    })

        return {
            "file_name": str(video_path.name),
            "total_count_sampled": len(detections_data),
            "detections": detections_data,
            "type": "video"
        }