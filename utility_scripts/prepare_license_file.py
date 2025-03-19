import os

from cryptography.fernet import Fernet
from pydantic import BaseModel
from datetime import datetime
from typing import Text

class LocalLicenseFile(BaseModel):
    customer_id: Text
    expiry: datetime
    activation_cutoff: datetime
    remaining_usage: int
    activated: bool = False

key = os.environ.get("ENCRYPTION_KEY")

message = LocalLicenseFile(
    customer_id="darme",
    expiry="2025-03-20T06:36:25.808364",
    activation_cutoff="2025-03-20T06:36:25.808364",
    remaining_usage=100,
    activated=False,
).model_dump_json()

with open("license.txt", "w") as file:
    file.write(Fernet(key.encode()).encrypt(message.encode()).decode())
