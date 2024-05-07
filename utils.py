import json
import os
import random
from typing import Any, List
from torch.utils.data import Dataset
from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import wandb
from PIL import Image
from PIL import ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

import pytorch_lightning as pl

class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        data_json: str,
        processor,
        model,
        max_patches: int,
        max_length:int,
        ignore_id: int = -100,
        task_start_token: str = "",
        prompt_end_token: str = None,
    ):
        super().__init__()

        self.dataset = data_json
        self.processor = processor
        self.added_tokens = []

        self.model = model
        self.max_patches = max_patches
        self.max_length = max_length
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

   
    
    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # prepare inputs
        # print(f"vqa mode on::{self.processor.image_processor.is_vqa}")
        image_folder_path = os.path.join(os.getenv("DATA_DIR"), "images")
        if not os.path.exists(image_folder_path):
            raise FileNotFoundError(f"Image folder not found: {image_folder_path}")
        img_name = f'{item["image_name"]}'
        img_path = os.path.join(image_folder_path, img_name)
        
        item["question"]=item["question"].replace("\u200f", "").strip()
        item["answer"]=item["answer"].replace("\u200f", "").strip()

        # img_path = os.path.join("/home/ubuntu/akshat/ara_intern_docvqa/master_compiled_images", doc_id)
        img = Image.open(img_path)
        encoding = self.processor(images=img, text = item["question"], max_patches=self.max_patches, return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # prepare targets
        # target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        target_sequence=item["answer"]
        input_ids = self.processor.tokenizer(
            target_sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels = input_ids.squeeze().clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        encoding["labels"] = labels
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return encoding, target_sequence
    
    


class Pix2Struct(pl.LightningModule):
    def __init__(self, config, processor, model, train_data, val_data):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def training_step(self, batch, batch_idx):
        encoding, _ = batch
        
        outputs = self.model(**encoding)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        encoding, answers = batch
        flattened_patches, attention_mask = encoding["flattened_patches"], encoding["attention_mask"]
        # batch_size = flattened_patches.shape[0]
        # we feed the prompt to the model
        # decoder_input_ids = torch.full((batch_size, 1), self.model.config.text_config.decoder_start_token_id, device=self.device)
        
        outputs = self.model.generate(flattened_patches=flattened_patches,
                                      attention_mask=attention_mask,
                                      # decoder_input_ids=decoder_input_ids,
                                      max_new_tokens=512,
                                      min_length = 1,
                                      return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
                
            # seq.replace("")
            # seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            # pred = re.sub(r"(?:(?<=>) | (?=", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "").strip()
            scores.append(edit_distance(pred.strip(), answer.strip()) / max(len(pred), len(answer), 1))
            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores)) 
        
        return scores

    def configure_optimizers(self):
        # optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=self.config.get("lr"), weight_decay=1e-05)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        # scheduler = get_cosine_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=self.config.get("warmup_epochs"),
        #                                             num_training_steps=self.config.get("max_epochs"))
        
        return [optimizer]

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.val_data