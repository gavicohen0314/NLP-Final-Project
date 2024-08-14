import numpy as np
import collections
import torch

class psudo_max_perplexity_DataCollator:
    """
    psudo maximum perplexity masking
    run the model n times on random masking  
    choose the masked sentence with highest perplexity (perplexity using loss not entropy)
    """

    def __init__(self, tokenizer, model, data_collator, n):
        self.tokenizer = tokenizer
        self.model = model
        self.data_collator = data_collator
        self.n = n


    def psudo_max_perplexity(self, features, tokenizer, data_collator, n, model):
        wwm_probability = 0.2

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


            # Variables to track the best masking configuration
            best_perplexity = float('-inf')
            best_masked_input_ids = None
            best_labels = None

            for _ in range(n):
                # Randomly mask words
                mask = np.random.binomial(1, wwm_probability, (len(mapping),))
                input_ids = torch.tensor(feature["input_ids"]).unsqueeze(0).to(device)  # Add batch dimension
                labels = input_ids.clone()  # Clone to avoid modifying the original
                new_labels = [-100] * len(labels)
                for word_id in np.where(mask)[0]:
                    word_id = word_id.item()
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
                
                # Calculate perplexity for this masked input
                with torch.no_grad():
                    outputs = model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()

                # Track the best masking based on perplexity
                if perplexity > best_perplexity:
                    best_perplexity = perplexity
                    best_masked_input_ids = input_ids
                    best_labels = new_labels

            # Update feature with the best masking configuration
            feature["input_ids"] = best_masked_input_ids
            feature["labels"] = best_labels

        return data_collator(features)


    def __call__(self, features):
        features = self.psudo_max_perplexity(features, self.tokenizer, self.data_collator, self.n, self.model)

        # Convert features to tensors
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        return batch



