import json
import pandas as pd

def preprocess_qa_data_to_plain_text_json(json_path='QA.json'):
    # Load the original JSON dataset
    with open(json_path, 'r') as f:
        data = json.load(f)['data']
    
    # Initialize list to store processed data
    processed_data = []

    for item in data:
        story = item['story']
        for q, a in zip(item['questions'], item['answers']):
            processed_data.append({
                "text": story,
                "question": q['input_text'],
                "answer": a['input_text']
            })
    
    # Save the processed data as a JSON file
    preprocessed_data_path = 'preprocessed_data_plain_text.json'
    with open(preprocessed_data_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print("Preprocessing completed. Data saved to", preprocessed_data_path)

if __name__ == "__main__":
    preprocess_qa_data_to_plain_text_json('./QA.json')


