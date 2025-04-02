import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import logging
import json
from typing import Dict, List
import sys

NUM_DASHES = 200

# Configure logging with both file and console handlers
def setup_logging(output_dir: str):
    """Setup logging configuration with both file and console handlers."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging("logs")

def setup_model_and_tokenizer(model_name: str = "distilgpt2"):
    """Setup the model and tokenizer with qLoRA configuration."""
    logger.info("\n" + "="*NUM_DASHES)
    logger.info(f"Setting up model and tokenizer for {model_name}")
    logger.info("-"*NUM_DASHES)
    
    # Configure 4-bit quantization
    logger.info("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer first
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Log initial tokenizer state
    logger.info("Initial tokenizer state:")
    logger.info(f"Padding token: {tokenizer.pad_token}")
    logger.info(f"Padding token ID: {tokenizer.pad_token_id}")
    logger.info(f"EOS token: {tokenizer.eos_token}")
    logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Set padding token to EOS token
    logger.info("Configuring padding token...")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Verify padding token configuration
    logger.info("Verifying padding token configuration...")
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.error("Failed to set padding token!")
        raise ValueError("Padding token configuration failed")
    
    logger.info("Final tokenizer state:")
    logger.info(f"Padding token: {tokenizer.pad_token}")
    logger.info(f"Padding token ID: {tokenizer.pad_token_id}")
    
    # Load model with padding token configuration
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Ensure model config has the correct padding token ID
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    
    # Verify model padding token configuration
    logger.info("Verifying model padding token configuration...")
    if model.config.pad_token_id != tokenizer.pad_token_id:
        logger.error("Model padding token ID mismatch!")
        raise ValueError("Model padding token configuration failed")
    
    logger.info("Final model state:")
    logger.info(f"Model padding token: {model.config.pad_token}")
    logger.info(f"Model padding token ID: {model.config.pad_token_id}")
    
    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    logger.info("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # GPT-2 specific attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\nModel Statistics:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    
    return model, tokenizer

def create_training_args(output_dir: str):
    """Create training arguments for GRPO."""
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Creating training arguments")
    logger.info("-"* NUM_DASHES)
    
    # Calculate compatible batch size and gradient accumulation steps
    num_generations = 2  # Number of generations per prompt
    effective_batch_size = 4  # Desired effective batch size
    per_device_batch_size = 2  # Batch size per device
    gradient_accumulation_steps = effective_batch_size // per_device_batch_size
    
    config = GRPOConfig(
        output_dir=output_dir,
        run_name=f"grpo-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        num_train_epochs=1,  # Reduced from 3 to 1
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",  # Changed from "epoch" to "no" to avoid model saving overhead
        eval_strategy="steps",  # Changed from "epoch" to "steps" for more frequent evaluation
        eval_steps=100,  # Evaluate every 100 steps
        load_best_model_at_end=False,  # Changed to False to reduce overhead
        report_to="tensorboard",
        max_prompt_length=256,  # Reduced from 512 to 256
        label_names=["input_ids", "attention_mask", "labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=num_generations,
        max_steps=500,  # Added maximum number of steps to limit training time
        warmup_steps=50,  # Added warmup steps
    )
    
    logger.info("Training Configuration:")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of epochs: {config.num_train_epochs}")
    logger.info(f"Maximum steps: {config.max_steps}")
    logger.info(f"Per device batch size: {config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"Number of generations per prompt: {config.num_generations}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Max prompt length: {config.max_prompt_length}")
    logger.info(f"Evaluation strategy: {config.eval_strategy} (every {config.eval_steps} steps)")
    logger.info(f"Save strategy: {config.save_strategy}")
    logger.info("="* NUM_DASHES + "\n")
    
    return config

def prepare_dataset():
    """Prepare the training and validation datasets."""
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Loading and preparing datasets")
    logger.info("-"* NUM_DASHES)
    
    logger.info("Loading TLDR dataset...")
    dataset = load_dataset("trl-lib/tldr")
    
    # Limit dataset size to reduce training time
    logger.info("Limiting dataset size to reduce training time...")
    max_train_samples = 1000  # Use only 1000 samples for training
    max_val_samples = 100     # Use only 100 samples for validation
    
    # Subsample the datasets
    train_dataset = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(max_val_samples, len(dataset["validation"]))))
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"Original training samples: {len(dataset['train']):,}")
    logger.info(f"Original validation samples: {len(dataset['validation']):,}")
    logger.info(f"Reduced training samples: {train_size:,}")
    logger.info(f"Reduced validation samples: {val_size:,}")
    logger.info("="* NUM_DASHES + "\n")
    
    return {
        "train": train_dataset,
        "validation": val_dataset
    }

def reward_len(completions, **kwargs):
    """Reward function that rewards completions close to 20 characters."""
    rewards = [-abs(20 - len(completion)) for completion in completions]
    return rewards

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,  # Reduced from 200 to 100 for more concise responses
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        min_length=10,  # Ensure responses have at least 10 tokens
        no_repeat_ngram_size=2  # Prevent repetition of bigrams
    )
    
    # Get the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (remove the prompt)
    response = generated_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    
    # Clean up the response by removing excessive newlines and whitespace
    response = ' '.join(response.split())
    
    # If response is empty, return a default response
    if not response:
        response = "Exercise benefits health."
    
    return response

def train_model(
    model_name: str = "distilgpt2",
    output_dir: str = "grpo_trained_model",
    num_prompts: int = 5
):
    """Main training function."""
    logger.info("Starting GRPO Training Process")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of test prompts: {num_prompts}")
    logger.info("-"* NUM_DASHES + "\n")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Create training arguments
    training_args = create_training_args(output_dir)
    
    # Prepare dataset
    datasets = prepare_dataset()
    
    # Verify padding token configuration before trainer initialization
    logger.info("Verifying padding token configuration before trainer initialization...")
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.error("Padding token not configured properly!")
        raise ValueError("Padding token configuration failed")
    
    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        reward_funcs=reward_len,
    )
    
    # Set the tokenizer in the trainer's processing class
    logger.info("Configuring trainer's processing class with tokenizer...")
    trainer.processing_class = tokenizer
    
    # Verify padding token configuration after trainer initialization
    logger.info("Verifying padding token configuration after trainer initialization...")
    if trainer.processing_class.pad_token is None or trainer.processing_class.pad_token_id is None:
        logger.error("Trainer processing class padding token not configured properly!")
        raise ValueError("Trainer processing class padding token configuration failed")
    
    # Test prompts
    test_prompts = [
        "Summarize the benefits of exercise in one sentence.",
        "What is the main idea of machine learning?",
        "Explain quantum computing briefly.",
        "What are the key features of artificial intelligence?",
        "Describe the process of photosynthesis in one line."
    ][:num_prompts]

    # Test before training
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Testing Model Before Training")
    logger.info("="* NUM_DASHES)
    before_results = {}
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nTest {i}/{len(test_prompts)}")
        logger.info(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        before_results[prompt] = response
        logger.info(f"Response: \"{response}\"")  # Add quotes for better visibility
        logger.info(f"Length: {len(response)} characters")
    logger.info("="* NUM_DASHES)

    # Train the model
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Starting Training Process")
    logger.info("="* NUM_DASHES)
    trainer.train()
    logger.info("Training completed successfully!")
    logger.info("="* NUM_DASHES)

    # Save the model
    logger.info("\nSaving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Test after training
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Testing Model After Training")
    logger.info("="* NUM_DASHES)
    after_results = {}
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nTest {i}/{len(test_prompts)}")
        logger.info(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        after_results[prompt] = response
        logger.info(f"Response: \"{response}\"")  # Add quotes for better visibility
        logger.info(f"Length: {len(response)} characters")
    logger.info("="* NUM_DASHES)

    # Save results
    logger.info("\nSaving test results...")
    results = {
        "before_training": before_results,
        "after_training": after_results
    }
    
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Print summary statistics
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Training Summary Statistics")
    logger.info("="* NUM_DASHES)
    avg_len_before = sum(len(v) for v in before_results.values()) / len(before_results)
    avg_len_after = sum(len(v) for v in after_results.values()) / len(after_results)
    logger.info(f"Average response length before training: {avg_len_before:.2f} characters")
    logger.info(f"Average response length after training: {avg_len_after:.2f} characters")
    logger.info(f"Target length: 20 characters")
    logger.info(f"Length reduction: {((avg_len_before - avg_len_after) / avg_len_before * 100):.2f}%")
    logger.info("="* NUM_DASHES + "\n")

    return results

if __name__ == "__main__":
    logger.info("\n" + "="* NUM_DASHES)
    logger.info("Starting GRPO Training Script")
    logger.info("="* NUM_DASHES)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    logger.info("-"* NUM_DASHES + "\n")
    
    try:
        results = train_model(
            model_name="distilgpt2",
            output_dir="grpo_trained_model",
            num_prompts=5
        )
        logger.info("Script completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise