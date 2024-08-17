import torch
import math
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler, default_data_collator, AutoModelForMaskedLM
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

class CustomTrainer:

  def __init__(self, model_checkpoint, tokenizer, device, dataset, data_collator, batch_size=64, num_train_epochs=3):
    self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint).to(device)
    self.tokenizer = tokenizer
    self.device = device
    self.dataset = dataset
    self.data_collator = data_collator
    self.batch_size = batch_size
    self.num_train_epochs = num_train_epochs


  def _insert_mask(self, batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    #masked_inputs = whole_word_masking_data_collator(features)
    masked_inputs = self.data_collator(features, self.model, self.tokenizer, self.device)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.cpu().numpy() for k, v in masked_inputs.items()}

  def train(self):
    eval_dataset = self.dataset["test"].map(
    self._insert_mask,
    batched=True,
    remove_columns=self.dataset["test"].column_names,
    )

    eval_dataset = eval_dataset.rename_columns(
      {
          "masked_input_ids": "input_ids",
          "masked_attention_mask": "attention_mask",
          "masked_labels": "labels",
      }
    )

    train_dataloader = DataLoader(
        self.dataset["train"],
        shuffle=True,
        batch_size=self.batch_size,
        collate_fn=self.data_collator,
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=self.batch_size, collate_fn=default_data_collator
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

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(self.batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
      
