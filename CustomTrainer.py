import torch
import math
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler, default_data_collator, AutoModelForMaskedLM
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

class CustomTrainer:

  def __init__(self, model, device, train_dataset, eval_dataset, data_collator, batch_size=64, num_train_epochs=3):
    self.model = model
    self.device = device
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.data_collator = data_collator
    self.batch_size = batch_size
    self.num_train_epochs = num_train_epochs
    self.train_losses = []  
    self.eval_losses = []

  def train(self):
    train_dataloader = DataLoader(
        self.train_dataset,
        shuffle=True,
        batch_size=self.batch_size,
        collate_fn=self.data_collator,
    )

    eval_dataloader = DataLoader(
        self.eval_dataset, batch_size=self.batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(self.model.parameters(), lr=5e-5)

    accelerator = Accelerator()

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        self.model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = self.num_train_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    # Pre-Training Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.inference_mode():
            outputs = model(**batch)

        loss = outputs.loss
        self.eval_losses.append(loss.item())

        losses.append(accelerator.gather(loss.repeat(self.batch_size)))
    
    losses = torch.cat(losses)
    losses = losses[: len(self.eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Pre-Training Perplexity: {perplexity}")

    for epoch in range(num_train_epochs):
        #training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            self.train_losses.append(loss.item())
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.inference_mode():
                outputs = model(**batch)

            loss = outputs.loss
            self.eval_losses.append(loss.item())

            losses.append(accelerator.gather(loss.repeat(self.batch_size)))
        
        losses = torch.cat(losses)
        losses = losses[: len(self.eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
