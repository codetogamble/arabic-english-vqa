import json
from typing import List
from fastapi import FastAPI, Form, UploadFile, File
from transformers import Pix2StructForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import os
import torch

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.model_path = os.getenv("MODEL_PATH")
    app.processor = AutoProcessor.from_pretrained(app.model_path)
    app.model = Pix2StructForConditionalGeneration.from_pretrained(app.model_path, is_encoder_decoder=True).to(app.device)

@app.post("/qna")
async def qna(image: UploadFile = File(...), question: List[str] = Form(...)):
    try:
        with open(os.path.join(app.model_path, "meta.json")) as f:
            config = json.load(f)
        max_patches = config.get("max_patches")
        max_length = config.get("max_length")
        if not max_patches or not max_length:
            return {"error": "Invalid model configuration"}
        final_response = []
        image = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        for q in question:
            inputs = app.processor(image, q, return_tensors="pt", max_patches=max_patches, max_length=max_length, font_path = "/usr/src/app/Arial.TTF").to(app.device)
            gen_tokens = app.model.generate(**inputs)
            output = app.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
            final_response.append({
                "question": q,
                "answer": output
            })
        
        return final_response
    except Exception as e:
        return {"error": str(e)}
