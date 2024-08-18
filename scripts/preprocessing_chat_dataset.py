import os
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_filter_dataset(dataset_name, language, sample_size):
    """
    Load and filter the dataset based on language and sample size.
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    print(f"Filtering for {language} language")
    filtered_dataset = dataset.filter(lambda example: example['language'] == language)
    
    print(f"Selecting {sample_size} samples")
    sampled_dataset = filtered_dataset.select(range(sample_size))
    
    return sampled_dataset

def format_conversation(conversation, tokenizer):
    """
    Format a single conversation with proper prefixes and end token.
    """
    formatted_conv = ""
    for turn in conversation:
        if turn['role'] == 'user':
            formatted_conv += f"H: {turn['content'].strip()}\n"
        elif turn['role'] == 'assistant':
            if not turn['content'].strip():
                return None  # Skip conversations with empty assistant responses
            formatted_conv += f"A: {turn['content'].strip()}\n"
    
    return formatted_conv.strip() + tokenizer.eos_token

def preprocess_conversations(examples, tokenizer):
    """
    Preprocess and tokenize conversations.
    """
    processed_conversations = [
        format_conversation(conv, tokenizer) 
        for conv in examples['conversation']
    ]
    processed_conversations = [conv for conv in processed_conversations if conv]
    
    return tokenizer(processed_conversations, padding=True, truncation=True, return_tensors="pt")

def main():
    # Configuration
    DATASET_NAME = "lmsys/lmsys-chat-1m"
    LANGUAGE = "English"
    SAMPLE_SIZE = 200000
    MODEL_NAME = "state-spaces/mamba-2.8b-hf"
    OUTPUT_DIR = "filtered_chat_dataset"

    # Load and prepare tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Load and filter dataset
    dataset = load_and_filter_dataset(DATASET_NAME, LANGUAGE, SAMPLE_SIZE)

    # Preprocess dataset
    print("Preprocessing dataset")
    preprocessed_dataset = dataset.map(
        lambda examples: preprocess_conversations(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Filter out empty conversations
    print("Filtering out empty conversations")
    filtered_dataset = preprocessed_dataset.filter(lambda example: len(example["input_ids"]) > 0)

    # Save processed dataset
    print(f"Saving processed dataset to: {OUTPUT_DIR}")
    filtered_dataset.save_to_disk(OUTPUT_DIR)

    # Print statistics and samples
    print_dataset_info(dataset, filtered_dataset)

def print_dataset_info(original_dataset, processed_dataset):
    print("\n" + "="*50 + "\n")
    print("Sample from original dataset:")
    print(original_dataset['train'][0])

    print("\n" + "="*50 + "\n")
    print("Sample from processed dataset (detokenized):")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    detokenized = gpt2_tokenizer.decode(processed_dataset['train'][0]['input_ids'])
    print(detokenized)

    print("\n" + "="*50 + "\n")
    print(f"Original dataset size: {len(original_dataset['train'])}")
    print(f"Processed dataset size: {len(processed_dataset['train'])}")

if __name__ == "__main__":
    main()