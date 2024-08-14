import numpy as np
import collections



class random_mask_DataCollator:
    """
    whole word random maksing
    """
    
    def __init__(self, tokenizer, data_collator):
        self.tokenizer = tokenizer
        self.data_collator = data_collator


    def random_mask(self, features, tokenizer, data_collator):
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

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels

        return data_collator(features)
    
    def __call__(self, features):
        features = self.random_mask(features, self.tokenizer, self.data_collator)

        # Convert features to tensors
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        return batch

