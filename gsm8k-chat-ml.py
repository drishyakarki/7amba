```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import re
from tqdm import tqdm

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to('cuda')
    return model, tokenizer

def generate_answers(model, tokenizer, prompts, max_new_tokens=100):
    chat_template = """
<|im_start|>human
{prompt}
<|im_end|>
<|im_start|>assistant
"""
    inputs = [chat_template.format(prompt=prompt) for prompt in prompts]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to('cuda')

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract only the assistant's response
    answers = [re.split(r'<\|im_start\|>assistant\n', output)[-1].split('<|im_end|>')[0].strip() for output in decoded_outputs]
    return answers

def extract_answer(text):
    match = re.search(r'\b\d+\b', text)
    return int(match.group()) if match else None

def evaluate_gsm8k(model, tokenizer, dataset, batch_size=16):
    correct = 0
    total_time = 0
    total_samples = len(dataset['test'])

    for i in tqdm(range(0, total_samples, batch_size)):
        batch = dataset['test'][i:i+batch_size]
        questions = batch['question']
        prompts = [f"Solve the following math problem step by step:\n\n{q}\n\nSolution:" for q in questions]

        start_time = time.time()
        responses = generate_answers(model, tokenizer, prompts)
        end_time = time.time()

        predicted_answers = [extract_answer(response) for response in responses]
        correct_answers = [extract_answer(answer) for answer in batch['answer']]

        correct += sum(p == c for p, c in zip(predicted_answers, correct_answers) if p is not None and c is not None)
        total_time += end_time - start_time

        for q, p, c in zip(questions, predicted_answers, correct_answers):
            print(f"Question: {q}")
            print(f"Model's answer: {p}")
            print(f"Correct answer: {c}")
            print("--------------------")

    accuracy = correct / total_samples
    avg_time = total_time / total_samples
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average time per question: {avg_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total samples processed: {total_samples}")

if __name__ == "__main__":
    model_name = "model"
    model, tokenizer = load_model_and_tokenizer(model_name)

    dataset = load_dataset("gsm8k", "main")

    evaluate_gsm8k(model, tokenizer, dataset)
```