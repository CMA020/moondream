import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from safetensors.torch import save_file
import os
from PIL import Image
import random

from tqdm import tqdm
from bitsandbytes.optim import AdamW8bit
# import wandb

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden, _lm_head, TextConfig

# Configuration
MODEL_PATH = "/content/moondream/models/model.safetensors"  # Path to your Moondream model weights
ANSWER_EOS = "<|endoftext|>"
LR = 3e-6
EPOCHS = 3
# GRAD_ACCUM_STEPS = 128
GRAD_ACCUM_STEPS = 128


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def text_loss(
    inputs_embeds: torch.Tensor, w: nn.Module, labels: torch.Tensor, config: TextConfig
):
    _, q_len, _ = inputs_embeds.shape
    hidden_BTC = _produce_hidden(inputs_embeds, w, config)
    lm_logits = _lm_head(hidden_BTC, w)

    loss = None
    if labels is not None:
        _, _, l_len = labels.shape
        shift_index = (q_len - l_len) - 1
        shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
        shifted_labels = labels.contiguous()
        loss = nn.CrossEntropyLoss()(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )
    return loss


class GoalDetectionDataset(Dataset):
    def __init__(self, yes_folder, no_folder, split="train", train_ratio=0.98):
        """
        Dataset for fine-tuning Moondream to detect if a ball is in the goal
        
        Args:
            yes_folder (str): Path to folder containing images where ball is in goal
            no_folder (str): Path to folder containing images where ball is not in goal
            split (str): 'train' or 'val' split
            train_ratio (float): Ratio of data to use for training vs validation
        """
        self.question = "\n\nQuestion: Is the ball in the goal?\n\nAnswer:"
        self.eos_token = ANSWER_EOS
        
        # Get all image files from both folders
        yes_images = [(os.path.join(yes_folder, f), "Yes") 
                      for f in os.listdir(yes_folder) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        no_images = [(os.path.join(no_folder, f), "No") 
                     for f in os.listdir(no_folder) 
                     if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        # Combine and shuffle the data
        all_data = yes_images + no_images
        random.seed(42)  # For reproducibility
        random.shuffle(all_data)
        
        # Split into train and validation
        split_idx = int(len(all_data) * train_ratio)
        
        if split == "train":
            self.data = all_data[:split_idx]
        else:  # "val" or any other value
            self.data = all_data[split_idx:]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, answer = self.data[idx]
        
        # Load and convert image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder black image in case of error
            image = Image.new('RGB', (224, 224), color=0)
            
        return {
            "image": image,
            "qa": {
                "question": self.question,
                "answer": f"{answer}{self.eos_token}",
            },
        }


def main():
    # Set device
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print(torch.device("cuda" if torch.cuda.is_available() else "cpu"), "deviiiiiiiiiiice")
    # elif torch.backends.mps.is_available():
    #     torch.set_default_device("mps")

    # Initialize wandb for experiment tracking
    # wandb.init(
    #     project="moondream-goal-detection",
    #     config={
    #         "EPOCHS": EPOCHS,
    #         "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
    #         "LR": LR,
    #     },
    # )

    # Load model and configuration
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    # Initialize optimizer
    optimizer = AdamW8bit(
        [
            {"params": model.text.parameters()},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    # Replace with your folder paths
    yes_folder = "/content/classifier_data/goal_crop_refined3"  
    no_folder = "/content/classifier_data/nogoal_crop_refined3"
    
    # Create datasets
    train_dataset = GoalDetectionDataset(yes_folder, no_folder, split="train")
    val_dataset = GoalDetectionDataset(yes_folder, no_folder, split="val")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Calculate total steps
    total_steps = EPOCHS * len(train_dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    # Training loop
    i = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for sample in train_dataset:
            i += 1
            
            # Get image embeddings
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
                
            # Get BOS token embedding
            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
                model.text,
            )
            
            # Get question embeddings
            question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            
            # Get answer embeddings
            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            
            # Combine all embeddings
            inputs_embeds = torch.cat(
                [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
            )
            
            # Calculate loss
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=torch.tensor([[answer_tokens]], device=model.device),
                config=config.text,
            )
            
            # Backward pass
            loss.backward()
            epoch_losses.append(loss.item())

            # Update weights after accumulating gradients
            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Update learning rate
                lr = lr_schedule(i // GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                    
                # Update progress bar
                pbar.set_postfix({
                    "epoch": epoch + 1, 
                    "step": i // GRAD_ACCUM_STEPS, 
                    "loss": loss.item()
                })
                pbar.update(1)
                
                # Log to wandb
                # wandb.log({
                #     "loss/train": loss.item(), 
                #     "lr": optimizer.param_groups[0]["lr"]
                # })
        
        # Calculate and log epoch average loss
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Training Loss: {avg_epoch_loss:.4f}")
        # wandb.log({"loss/epoch": avg_epoch_loss})
        
        # Validation after each epoch
        if len(val_dataset) > 0:
            model.eval()
            val_losses = []
            
            for sample in val_dataset:
                with torch.no_grad():
                    # Get image embeddings
                    img_emb = model._run_vision_encoder(sample["image"])
                    
                    # Get BOS token embedding
                    bos_emb = text_encoder(
                        torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
                        model.text,
                    )
                    
                    # Get question embeddings
                    question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
                    question_emb = text_encoder(
                        torch.tensor([[question_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)
                    
                    # Get answer embeddings
                    answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
                    answer_emb = text_encoder(
                        torch.tensor([[answer_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)
                    
                    # Combine all embeddings
                    inputs_embeds = torch.cat(
                        [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
                    )
                    
                    # Calculate validation loss
                    val_loss = text_loss(
                        inputs_embeds=inputs_embeds,
                        w=model.text,
                        labels=torch.tensor([[answer_tokens]], device=model.device),
                        config=config.text,
                    )
                    val_losses.append(val_loss.item())
            
            # Calculate and log validation metrics
            avg_val_loss = sum(val_losses) / len(val_losses)
            # wandb.log({"loss/val": avg_val_loss})
            print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}")
            
    # Finish logging
    # wandb.finish()
    
    # Save the fine-tuned model
    save_file(
        model.state_dict(),
        "/content/drive/MyDrive/moon_weigths/moondream_goal_detection.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_text
    """
    main()