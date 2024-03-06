import json
import os
from utils import ImageCaptioningDataset, Pix2Struct
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
# dataset = load_dataset("naver-clova-ix/cord-v2")



# example = dataset['train'][0]
# image = example['image']
# let's make the image a bit smaller when visualizing
# width, height = image.size


# let's load the corresponding JSON dictionary (as string representation)
# ground_truth = example['ground_truth']
# print(ground_truth)

# from ast import literal_eval

# literal_eval(ground_truth)['gt_parse']


from transformers import Pix2StructForConditionalGeneration, AutoProcessor

# repo_id = "google/pix2struct-docvqa-base"
# repo_id = "/home/ubuntu/akshat/finetuning/latest_model_4_75"
repo_id = "/home/ubuntu/pix2struct_finetining/sidharth16_finetuned_endocqa_finetuned_on_all_arabic_data_conti"
# repo_id = "docvqa_v4_17th_nov_10_epochs"
processor = AutoProcessor.from_pretrained(repo_id)
model = Pix2StructForConditionalGeneration.from_pretrained(repo_id, is_encoder_decoder=True)
processor.image_processor.is_vqa = True


# with open("/home/ubuntu/pix2stuct_finetuning_project/ara_intern_docvqa/final_train_listdata.json", "r") as f:
#       train_json_list = json.load(f)
# with open("/home/ubuntu/pix2stuct_finetuning_project/ara_intern_docvqa/final_val_listdata.json", "r") as f:
#       test_json_list = json.load(f)

# with open("/home/ubuntu/pix2stuct_finetuning_project/16th_nov_data/intern_data/final_train_shortlong.json", "r") as f:
#       train_json_sl= json.load(f)
# with open("/home/ubuntu/pix2stuct_finetuning_project/16th_nov_data/intern_data/final_val_shortlong.json", "r") as f:
#       test_json_sl = json.load(f)
      
# with open("/home/ubuntu/pix2stuct_finetuning_project/16th_nov_data/hrsd/hrsd_looking_documents_train.json", "r") as f:
#       train_json_hrsd= json.load(f)
# with open("/home/ubuntu/pix2stuct_finetuning_project/16th_nov_data/hrsd/hrsd_looking_documents_val.json", "r") as f:
#       test_json_hrsd = json.load(f)
# train_json = train_json_list+train_json_sl+train_json_hrsd
# test_json = test_json_list+test_json_sl+test_json_hrsd
with open("/home/ubuntu/pix2struct_finetining/invest_bank_data/data_sep_train.json", "r") as f:
      train_json_edc_en= json.load(f)
with open("/home/ubuntu/pix2struct_finetining/invest_bank_data/data_sep_test.json", "r") as f:
      test_json_edc_en = json.load(f)
      
# with open("/home/ubuntu/pix2struct_finetining/data/all_jsons/rta2_train_data.json", "r") as f:
#       train_json_edc_ar_en= json.load(f)
# with open("/home/ubuntu/pix2struct_finetining/data/all_jsons/rta2_test_data.json", "r") as f:
#       test_json_edc_ar_en = json.load(f)
      
train_json = train_json_edc_en + test_json_edc_en
test_json = train_json 

random.shuffle(train_json)
random.shuffle(test_json)
train_dataset = ImageCaptioningDataset(train_json, processor, model,
                                       split="train", sort_json_key=False) # cord dataset is preprocessed, so no need for this
val_dataset = ImageCaptioningDataset(test_json, processor, model,
                                       split="validation", sort_json_key=False) # cord dataset is preprocessed, so no need for this
     
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


# trainer.fit(pl_module, ckpt_path = "/home/ubuntu/akshat/finetuning/outputs/aradocvqa-epoch=10-train_loss=0.01.ckpt")
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