import numpy as np
import collections
import torch
import torch.nn.functional as F


class max_loss_no_mask_DataCollator:
    """
    Run model on sequence without masking,
    check which words had the highest loss,
    and mask those tokens.
    """

    def __init__(self, tokenizer, model, data_collator):
        self.tokenizer = tokenizer
        self.model = model
        self.data_collator = data_collator

    def max_loss_no_mask(self, features, tokenizer, data_collator, model):
        device = model.device
        mask_percentage = 0.2  # Mask the top 20% of tokens based on loss

        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None

            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            input_ids = torch.tensor(feature["input_ids"]).unsqueeze(0).to(device)  # Add batch dimension
            labels = input_ids.clone()

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            # Calculate the per-token loss for each token in the sequence
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

            # Flatten the logits and labels to compute cross-entropy for each token
            losses = loss_fct(
                logits.view(-1, logits.size(-1)),  # Flatten logits
                labels.view(-1)                    # Flatten labels
            )

            # Reshape the losses to match the original sequence length
            per_token_loss = losses.view(input_ids.size(0), -1).squeeze()

            # Determine the number of tokens to mask
            num_tokens_to_mask = max(1, int(len(per_token_loss) * mask_percentage))

            # Get the indices of the top loss tokens
            top_loss_indices = torch.topk(per_token_loss, num_tokens_to_mask).indices

            # Mask the selected tokens
            new_labels = [-100] * len(labels.squeeze())
            input_ids = input_ids.squeeze().tolist()
            for idx in top_loss_indices:
                idx = idx.item()  # Convert to standard Python int
                new_labels[idx] = labels[0][idx].item()
                input_ids[idx] = tokenizer.mask_token_id
            feature["input_ids"] = input_ids
            feature["labels"] = new_labels

        return data_collator(features)




    def __call__(self, features):
        features = self.max_loss_no_mask(features, self.tokenizer, self.data_collator, self.model)

        # Convert features to tensors
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        return batch