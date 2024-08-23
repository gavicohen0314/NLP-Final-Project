from transformers import DataCollatorForLanguageModeling

class random_collator:
     
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
          
    def __call__(self, features):
        return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
    
