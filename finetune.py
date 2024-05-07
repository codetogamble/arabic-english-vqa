import json
import os
from utils import ImageCaptioningDataset, Pix2Struct
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
from transformers import Pix2StructForConditionalGeneration, AutoProcessor


repo_id = os.getenv("MODEL_PATH")
processor = AutoProcessor.from_pretrained(repo_id)
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
    train_data = json.load(f)

with open(test_file_path, 'r') as f:
    test_data = json.load(f)
      
train_json = train_data
test_json = test_data 

random.shuffle(train_json)
random.shuffle(test_json)

max_patches = int(os.getenv('MAX_PATCHES', 3584))
max_length = int(os.getenv('MAX_LENGTH', 256))

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


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

# get first batch
batch = next(iter(train_dataloader))
encoding, target_sequences = batch

for k,v in encoding.items():
      print(k,v.shape)
print(processor.batch_decode([id for id in encoding["labels"].squeeze().tolist() if id != -100]))

lr_logger = LearningRateMonitor(logging_interval='step')
config = {
          "num_warmup_epochs": 0,
          "max_epochs": 8,
          "lr": 5e-5,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "warmup_steps": 0, # 800/8*30/10, 10%
          "accumulate_grad_batches": 8,
          "verbose": True,
          }

pl_module = Pix2Struct(config, processor, model, train_dataloader, val_dataloader)



wandb_logger = WandbLogger(project="Pix2Struct", name="demo-run-pix2struct-adafactor")

checkpoint_callback = ModelCheckpoint(
    save_top_k=100,
    monitor="train_loss",
    mode="min",
    every_n_train_steps = 50,
    dirpath="/home/ubuntu/pix2stuct_finetuning_project/finetuning/outputs",
    filename="aradocvqa-{epoch:02d}-{train_loss:.2f}",
)
import pytorch_lightning as pl

trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs = config.get("max_epochs"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"), # use gradient clipping
        accumulate_grad_batches=config.get("accumulate_grad_batches"), # use gradient accumulation
        logger=wandb_logger,
        callbacks = [lr_logger],
        log_every_n_steps = 2,
        precision = 16
)


trainer.fit(pl_module)


model_output_name = "invest_bank_v5_9feb_no_n"
# model_output_name = "english_base_finetuned_on_edc_data_24nov_en"
try:
      pl_module.model.save_pretrained(model_output_name)
except Exception as e:
      print("saving og")
      model.save_pretrained(model_output_name)
processor.save_pretrained(model_output_name)
processor.tokenizer.save_pretrained(model_output_name)

with open("invest_bank_v5_9feb_no_n/meta.json", "w") as f:
      json.dump({
            "max_patches": 3072,
            "max_tokens": 256,
            "train_date": "09/02/2024",
      }, f, indent=4, ensure_ascii=False)