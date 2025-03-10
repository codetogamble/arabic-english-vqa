import json
from typing import List
from fastapi import FastAPI, Form, UploadFile, File
from pydantic import BaseModel
from transformers import Pix2StructForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import os
import torch
import pdf2image
import base64
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
            gen_tokens = app.model.generate(**inputs,  max_new_tokens=max_length, return_dict_in_generate=True, output_scores=True, num_beams=1)
            output = app.processor.batch_decode(gen_tokens.sequences, skip_special_tokens=True)[0]
            
            word_level_scores = torch.stack(gen_tokens.scores, dim=1)
            soft_m = torch.nn.functional.softmax(word_level_scores, dim=2)
            conf = soft_m.max(dim=2).values
            till_where = torch.where(gen_tokens.sequences == 1)[1]
            mean_confs = []
            first_conf = []
            second_conf = []
            try:
                for c, w in zip(conf, till_where):
                    mean_confs.append(float(torch.mean(c[0 : int(w)])))
                    first_conf.append(c[0].item())
                    second_conf.append(c[1].item())
                confidence = {
                    "mean_confidence": mean_confs[0],
                    "first_confidence": first_conf[0],
                    "second_confidence": second_conf[0],
                }
            except Exception as e:
                print(e)
                confidence = {
                    "mean_confidence": 0,
                    "first_confidence": 0,
                    "second_confidence": 0,
                }
            
            final_response.append({
                "question": q,
                "answer": output,
                "confidence": confidence
            })
        
        return final_response
    except Exception as e:
        return {"error": str(e)}

class PdfRequest(BaseModel):
    pdf: str
    questions: List[str]

@app.post("/json/qna/pdf")
async def qna(pdf_request: PdfRequest):
    pdf = base64.b64decode(pdf_request.pdf.strip())
    questions = pdf_request.questions
    try:
        with open(os.path.join(app.model_path, "meta.json")) as f:
            config = json.load(f)
        max_patches = config.get("max_patches")
        max_length = config.get("max_length")
        if not max_patches or not max_length:
            return {"error": "Invalid model configuration"}
        final_response = []
        images = pdf2image.convert_from_bytes(pdf)
        for i, image in enumerate(images):
            image = image.convert("RGB")
            for q in questions:
                inputs = app.processor(image, q, return_tensors="pt", max_patches=max_patches, max_length=max_length, font_path = "/usr/src/app/Arial.TTF").to(app.device)
                gen_tokens = app.model.generate(**inputs,  max_new_tokens=max_length, return_dict_in_generate=True, output_scores=True, num_beams=1)
                output = app.processor.batch_decode(gen_tokens.sequences, skip_special_tokens=True)[0]
                
                word_level_scores = torch.stack(gen_tokens.scores, dim=1)
                soft_m = torch.nn.functional.softmax(word_level_scores, dim=2)
                conf = soft_m.max(dim=2).values
                till_where = torch.where(gen_tokens.sequences == 1)[1]
                mean_confs = []
                first_conf = []
                second_conf = []
                try:
                    for c, w in zip(conf, till_where):
                        mean_confs.append(float(torch.mean(c[0 : int(w)])))
                        first_conf.append(c[0].item())
                        second_conf.append(c[1].item())
                    confidence = {
                        "mean_confidence": mean_confs[0],
                        "first_confidence": first_conf[0],
                        "second_confidence": second_conf[0],
                    }
                except Exception as e:
                    print(e)
                    confidence = {
                        "mean_confidence": 0,
                        "first_confidence": 0,
                        "second_confidence": 0,
                    }
                
                final_response.append({
                    "question": q,
                    "answer": output,
                    "confidence": confidence,
                    "page": i
                })
        
        return final_response
    except Exception as e:
        return {"error": str(e)}