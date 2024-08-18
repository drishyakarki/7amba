from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_function(examples):
    conversations = examples['conversation']
    processed_conversations = []
    
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf") 
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    
    for conv in conversations:
        processed_conv = ""
        skip_conversation = False
        for turn in conv:
            if turn['role'] == 'user':
                processed_conv += f"H: {turn['content'].strip()}\n"
            elif turn['role'] == 'assistant':
                if not turn['content'].strip():  # Check if the assistant's response is empty
                    skip_conversation = True
                    break
                processed_conv += f"A: {turn['content'].strip()}\n"
        
        if not skip_conversation:
            processed_conv += tokenizer.eos_token  # End token at the end of each conversation
            processed_conversations.append(processed_conv.strip())
    
    tokenized = tokenizer(processed_conversations, padding=True, truncation=True, return_tensors="pt")
    
    return tokenized

# Load the dataset
ds = load_dataset("lmsys/lmsys-chat-1m")
ds = ds.filter(lambda example: example['language'] == 'English')
ds = ds.select(range(200000))

# Apply preprocessing
preprocessed_ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

filtered_ds = preprocessed_ds.filter(lambda example: len(example["input_ids"]) > 0)
filtered_ds.save_to_disk("filtered_chat_dataset")

print("\n" + "="*50 + "\n")
print("First data in the original dataset:")
print(ds['train'][0])

print("\n" + "="*50 + "\n")
print("First data in the filtered dataset (detokenized):")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
detokenized = tokenizer.decode(filtered_ds['train'][0]['input_ids'])
print(detokenized)

print("\n" + "="*50 + "\n")
print(f"Original dataset size: {len(ds['train'])}")

print("\n" + "="*50 + "\n")
print(f"Filtered dataset size: {len(filtered_ds['train'])}")