import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    """Preprocess text for a more lenient comparison."""
    return text.lower().strip().replace(" ", "").replace(".", "").replace(",", "")

def evaluate_metrics(predictions, references):
    """Evaluates precision, recall, and F1 score based on cleaned text matches."""
    y_true = [1 if pred == ref else 0 for pred, ref in zip(predictions, references)]
    y_pred = [1 if pred in ref or ref in pred else 0 for pred, ref in zip(predictions, references)]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return precision, recall, f1

def evaluate(data, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions, true_answers = [], []
    
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
        question, context, true_answer = row['question'], row['text'], row['answer']
        input_ids = tokenizer.encode(question, context, truncation=True, max_length=512, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        sep_index = input_ids[0].tolist().index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = input_ids.size(1) - num_seg_a
        segment_ids = torch.tensor([[0]*num_seg_a + [1]*num_seg_b], dtype=torch.long)
        
        input_ids, segment_ids = input_ids.to(device), segment_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=segment_ids)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        if answer_end > answer_start:
            answer_tokens = tokens[answer_start:answer_end]
        else:
            answer_tokens = tokens[answer_start:answer_start+1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        
        pred_answer_clean = clean_text(answer)
        true_answer_clean = clean_text(true_answer)
        predictions.append(pred_answer_clean)
        true_answers.append(true_answer_clean)
    
    precision, recall, f1 = evaluate_metrics(predictions, true_answers)
    accuracy = sum(1 for pred, true in zip(predictions, true_answers) if pred == true) / len(data)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main():
    data_path = "https://res.cloudinary.com/doec3busn/raw/upload/v1709285138/CoQA_data_t0zild.csv"
    data = load_data(data_path)
    
    print(f"Number of question and answers: {len(data)}")
    
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    evaluate(data, model, tokenizer)

if __name__ == "__main__":
    main()
