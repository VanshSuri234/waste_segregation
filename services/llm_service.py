import os
from groq import Groq
import json
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile" 
        self.last_raw_data = None 

    def generate_segregation_report(self, batch_data, metrics):
        self.last_raw_data = {
            "detections": batch_data,
            "robot_metrics": metrics
        }
        
        # High-level technical header only
        prompt = f"Summarize detection of {len(batch_data)} files. Mention count and primary class. No fluff."
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a precise NTPC industrial auditor."},
                          {"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception:
            return "Batch data synchronized. Ready for audit queries."

    def ask_general_question(self, user_query):
        if not self.last_raw_data:
            return "Error: No active batch data. Please upload images to populate the auditor logic."

        # STRICT SYSTEM PROMPT FOR INDUSTRIAL ACCURACY
        system_instructions = f"""
        ROLE: NTPC SENIOR AI AUDITOR (Industrial Logic Engine)
        TONE: Assertive, Technical, Concise.
        
        CONTEXT DATA (JSON): {json.dumps(self.last_raw_data)}

        RESPONSE RULES:
        1. NO INTRODUCTIONS: Do not start with "Here is..." or "Based on the data...". Start immediately with the answer.
        2. TABLES: Use Markdown tables for ANY comparison or distribution list.
        3. SIZE LOGIC: Calculate Min/Max/Avg using 'area_px' from the 'detections' list.
        4. WEIGHT LOGIC: Use formula (Area_px * 0.00015 kg) for plastics/cardboard and (Area_px * 0.0004 kg) for metals/stones if exact density is missing.
        5. OVERLAP/SHAPE: Analyze the 'bbox' and 'center' coordinates. If centers are within 50px of each other, classify as "Overlapping/Touching".
        6. ACCURACY: Distinguish between System Accuracy (94% per metrics) and Model Confidence (from detection confidence scores).
        7. SENSOR FUSION: Explain that low visual confidence triggers ToF (depth) or Thermal sensors (per robot_metrics).
        8. DEFORMABLE/IRREGULAR: Identify objects with high aspect ratios or low confidence as "Deformable/Irregular Shape Risks".
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_query}
                ],
                model=self.model_name,
                temperature=0.0 # Zero creativity for data integrity
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ AUDIT ERROR: {str(e)}"