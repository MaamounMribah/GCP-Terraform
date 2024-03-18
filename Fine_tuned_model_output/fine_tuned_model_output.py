import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

class QADataset(Dataset):
    def __init__(self, tokenizer, data_file, max_len=512):
        self.data = pd.read_json(data_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        question, context = data_row['question'], data_row['text']
        encoding = self.tokenizer.encode_plus(question, context,
                                               add_special_tokens=True,
                                               max_length=self.max_len,
                                               return_token_type_ids=True,
                                               padding="max_length",
                                               truncation=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')

        # Dummy start and end positions, replace these with your actual method to find start and end positions
        start_positions = torch.tensor(0)
        end_positions = torch.tensor(1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'start_positions': start_positions,
            'end_positions': end_positions
        }

def train_model(model, tokenizer, train_data, device, epochs=3, batch_size=5):
    train_dataset = QADataset(tokenizer, data_file=train_data)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epochs)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch + 1}, Train loss: {total_train_loss / len(train_dataloader)}")


def load_and_prepare_data(file_path):
    data = pd.read_json(file_path)
    return data

def perform_inference(model, tokenizer, device, data):
    for index, row in data.iterrows():
        question, context = row['question'], row['text']
        inputs = tokenizer.encode_plus(context, question, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer_tokens = input_ids[0, answer_start:answer_end]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model.to(device)

    data_file = './preprocessed_data_plain_text.json'  # Adjust path as necessary
    data = load_and_prepare_data(data_file)
    
    # You might want to adjust sample size or select specific entries for demonstration
    perform_inference(model, tokenizer, device, data.sample(5))  # Here, sample(5) randomly selects 5 entries for inference
