import json
import os
from utils import ImageCaptioningDataset, LogValidationDistanceCallback, Pix2Struct
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from datetime import datetime

repo_id = os.getenv("MODEL_PATH")
processor = Pix2StructProcessor.from_pretrained(repo_id)
model = Pix2StructForConditionalGeneration.from_pretrained(repo_id, is_encoder_decoder=True)
processor.image_processor.is_vqa = True



data_dir = os.getenv('DATA_DIR')
train_file_path = os.path.join(data_dir, 'train.json')
test_file_path = os.path.join(data_dir, 'test.json')

if not os.path.exists(train_file_path):
      raise FileNotFoundError(f"File not found: {train_file_path}")
if not os.path.exists(test_file_path):
      raise FileNotFoundError(f"File not found: {test_file_path}")

with open(train_file_path, 'r') as f:
    train_json = json.load(f)

with open(test_file_path, 'r') as f:
    test_json = json.load(f)


random.shuffle(train_json)
random.shuffle(test_json)

max_patches = int(os.getenv('MAX_PATCHES', 3584))
max_length = int(os.getenv('MAX_LENGTH', 256))
batch_size = int(os.getenv('BATCH_SIZE', 1))
num_gpus = int(os.getenv('NUM_GPUS', 1))
num_epochs = int(os.getenv('~', 1))
lr = float(os.getenv('LR', 5e-5))

train_dataset = ImageCaptioningDataset(train_json, processor, model,
                                       max_patches=max_patches, max_length=max_length) 
val_dataset = ImageCaptioningDataset(test_json, processor, model,
                                       max_patches=max_patches, max_length=max_length) 
     
encoding, target_sequence = train_dataset[0]
print(encoding.keys())

print(processor.decode([id.item() for id in encoding["labels"] if id != -100]))


print(target_sequence)


print("Number of added tokens train:", len(train_dataset.added_tokens))
print(val_dataset.added_tokens)

print("Number of added tokens test:", len(train_dataset.added_tokens))
print(val_dataset.added_tokens)


len(processor.tokenizer)

from torch.utils.data import DataLoader


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

batch = next(iter(train_dataloader))
encoding, target_sequences = batch

for k,v in encoding.items():
      print(k,v.shape)
print(processor.batch_decode([id for id in encoding["labels"].squeeze().tolist() if id != -100]))

config = {
          "num_warmup_epochs": 0,
          "max_epochs": num_epochs,
          "lr": lr,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "warmup_steps": 0, # 800/8*30/10, 10%
          "accumulate_grad_batches": 8,
          "verbose": True,
          }

pl_module = Pix2Struct(config, processor, model, train_dataloader, val_dataloader)

score_file_path = os.path.join(os.getenv("DATA_DIR"), "training", "scores.txt")
os.makedirs(os.path.dirname(score_file_path), exist_ok=True)
if not os.path.exists(score_file_path):
      with open(score_file_path, "w") as f:
            f.write("edit_distance\n")

validation_logger = LogValidationDistanceCallback(score_file_path)



import pytorch_lightning as pl

trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        max_epochs = config.get("max_epochs"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"), # use gradient clipping
        accumulate_grad_batches=config.get("accumulate_grad_batches"), # use gradient accumulation
        callbacks = [validation_logger],
        log_every_n_steps = 2,
      #   precision = 16
)


trainer.fit(pl_module)


model_output_name = os.path.join(os.getenv("DATA_DIR"), "training", "model")
os.makedirs(model_output_name, exist_ok=True)
try:
      pl_module.model.save_pretrained(model_output_name)
except Exception as e:
      print("saving og")
      model.save_pretrained(model_output_name)
processor.save_pretrained(model_output_name)
processor.tokenizer.save_pretrained(model_output_name)
with open(os.path.join(model_output_name, "meta.json"), "w") as f:
      json.dump({
            "max_patches": max_patches,
            "max_length": max_length,
            "train_date": str(datetime.now())
      }, f, indent=4, ensure_ascii=False)