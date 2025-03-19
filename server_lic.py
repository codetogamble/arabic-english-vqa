import json
from typing import List
from fastapi import FastAPI, Form, UploadFile, File
from transformers import Pix2StructForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import os
import torch
from lic import _FILE_ENCRYPTION_KEY
from cryptography.fernet import Fernet
from datetime import datetime
from typing import Text
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, ConfigDict

app = FastAPI()


class LocalLicenseFile(BaseModel):
    customer_id: Text
    expiry: datetime
    activation_cutoff: datetime
    remaining_usage: int
    activated: bool = False


class LicenseResponse(BaseModel):
    customer_id: str
    is_valid: bool
    utc_time: datetime
    remaining_usage: int = 0


class CannotConnectToLicenseServer(Exception):
    pass


class DataExpired(Exception):
    pass


class InvalidPacket(Exception):
    pass


class LicenseActivationFailed(Exception):
    pass


class LicenseExpired(Exception):
    pass


class UsageLimitExceeded(Exception):
    pass


class LocalLicenseManager:
    def __init__(self, directory: Path) -> None:
        self.__FILE_ENCRYPTION_KEY = _FILE_ENCRYPTION_KEY
        self.directory = directory
        self.license_file = directory / "license.txt"
        self.usage_file = directory / "usage.json"
        assert self.license_file.exists(), "License file does not exist"

    @staticmethod
    def encrypt(message: str, key: str) -> str:
        return Fernet(key.encode()).encrypt(message.encode()).decode()

    @staticmethod
    def decrypt(data: str, key: str) -> str:
        return Fernet(key.encode()).decrypt(data.encode()).decode()
    def __check_activation(self, data: LocalLicenseFile):
        if not data.activated:
            if datetime.now(timezone.utc) < data.activation_cutoff:
                data.activated = True
                with open(self.license_file, "w") as file:
                    file.write(
                        self.encrypt(data.model_dump_json(), self.__FILE_ENCRYPTION_KEY)
                    )
            else:
                raise LicenseActivationFailed(
                    "License not activated and activation time expired"
                )

    def validate_license(self):
        with open(self.license_file) as file:
            data = file.read()
            decrypted = self.decrypt(data, self.__FILE_ENCRYPTION_KEY)

        data = LocalLicenseFile(**json.loads(decrypted))
        data.expiry = data.expiry.replace(tzinfo=timezone.utc)
        data.activation_cutoff = data.activation_cutoff.replace(tzinfo=timezone.utc)
        self.__check_activation(data)

        if datetime.now(timezone.utc) > data.expiry:
            raise LicenseExpired("License expired")

        if data.remaining_usage <= 0:
            raise UsageLimitExceeded("Usage limit exceeded")

        return LicenseResponse(
            customer_id=data.customer_id,
            is_valid=True,
            utc_time=datetime.now(timezone.utc),
            remaining_usage=data.remaining_usage,
        )

    def update_usage(self, usage_amount: int):
        with open(self.license_file) as file:
            data = file.read()
            decrypted = self.decrypt(data, self.__FILE_ENCRYPTION_KEY)

        data = LocalLicenseFile(**json.loads(decrypted))
        data.expiry = data.expiry.replace(tzinfo=timezone.utc)
        data.activation_cutoff = data.activation_cutoff.replace(tzinfo=timezone.utc)
        self.__check_activation(data)

        if datetime.now(timezone.utc) > data.expiry:
            raise LicenseExpired("License expired")

        if data.remaining_usage <= 0:
            raise UsageLimitExceeded("Usage limit exceeded")

        data.remaining_usage -= usage_amount
        with open(self.license_file, "w") as file:
            file.write(self.encrypt(data.model_dump_json(), self.__FILE_ENCRYPTION_KEY))

        with open(self.usage_file, "a") as file:
            file.write(
                json.dumps(
                    {
                        "usage_amount": usage_amount,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
            file.write("\n")


@app.on_event("startup")
async def startup_event():
    app.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.model_path = os.getenv("MODEL_PATH")
    app.license_path = os.getenv("LICENSE_PATH", "/usr/src/app/license")
    app.processor = AutoProcessor.from_pretrained(app.model_path)
    app.model = Pix2StructForConditionalGeneration.from_pretrained(
        app.model_path, is_encoder_decoder=True
    ).to(app.device)
    app.license_manager = LocalLicenseManager(Path(app.license_path))
    app.license_manager.validate_license()


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
            inputs = app.processor(
                image,
                q,
                return_tensors="pt",
                max_patches=max_patches,
                max_length=max_length,
                font_path="/usr/src/app/Arial.TTF",
            ).to(app.device)
            gen_tokens = app.model.generate(
                **inputs,
                max_new_tokens=max_length,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=1
            )
            output = app.processor.batch_decode(
                gen_tokens.sequences, skip_special_tokens=True
            )[0]

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

            final_response.append(
                {"question": q, "answer": output, "confidence": confidence}
            )
        usage = 1
        license_manager: LocalLicenseManager = app.license_manager
        license_manager.update_usage(usage)
        return final_response
    except Exception as e:
        return {"error": str(e)}


class TestUsageRequest(BaseModel):
    usage: int

@app.post("/test_lic")
async def update_lic(request: TestUsageRequest):
    usage = request.usage
    license_manager: LocalLicenseManager = app.license_manager
    license_manager.update_usage(usage)
    return {"status": "success"}