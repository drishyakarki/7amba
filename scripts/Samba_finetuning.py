import os
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

def load_model_and_tokenizer(model_name):
    """
    Load the pre-trained model and tokenizer.
    """
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def load_dataset(dataset_path):
    """
    Load the preprocessed dataset.
    """
    print(f"Loading dataset from: {dataset_path}")
    return load_dataset(dataset_path)

def configure_training_arguments(output_dir, epochs, batch_size, learning_rate):
    """
    Configure the training arguments.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        learning_rate=learning_rate
    )

def configure_lora(rank, target_modules):
    """
    Configure the LoRA (Low-Rank Adaptation) settings.
    """
    return LoraConfig(
        r=rank,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none"
    )

def setup_trainer(model, tokenizer, training_args, lora_config, dataset, text_field):
    """
    Set up the SFT (Supervised Fine-Tuning) Trainer.
    """
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset,
        dataset_text_field=text_field,
    )

def main():
    # Configuration
    MODEL_NAME = "state-spaces/mamba-2.8b-hf"
    DATASET_PATH = 'preprocessed_lm_sys_data/'
    OUTPUT_DIR = "./results"
    EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-3
    LORA_RANK = 8
    LORA_TARGET_MODULES = ["x_proj", "embeddings", "in_proj", "out_proj"]
    DATASET_TEXT_FIELD = "quote"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # Configure training arguments
    training_args = configure_training_arguments(OUTPUT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE)

    # Configure LoRA
    lora_config = configure_lora(LORA_RANK, LORA_TARGET_MODULES)

    # Set up trainer
    trainer = setup_trainer(model, tokenizer, training_args, lora_config, dataset, DATASET_TEXT_FIELD)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    main()