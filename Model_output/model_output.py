import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load the pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, context):
    # Encode the question-context pair
    input_ids = tokenizer.encode(question, context)
    # Convert IDs to tokens list
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Define segment ids based on the location of [SEP] token
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # Ensure every input token has a segment ID
    assert len(segment_ids) == len(input_ids)

    # Model's output
    outputs = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    # Identify the tokens corresponding to the start and end of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 to include the end token

    # Combine tokens for the answer and handle ## subwords
    answer = tokens[answer_start]  # Start with the first token.
    for i in range(answer_start + 1, answer_end):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return answer

# Example usage
context = "The Matrix is a 1999 science fiction action film written and directed by the Wachowskis. It stars Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano."
question = "What impact has elon musk had on the tech industry??"

answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")