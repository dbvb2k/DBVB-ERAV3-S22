import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import wandb
from typing import List, Dict
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_wandb_api_key():
    """Check if WANDB_API_KEY is set in environment variables."""
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        logger.warning("WANDB_API_KEY environment variable is not set!")
        logger.warning("Please set it using: export WANDB_API_KEY=your_api_key")
        return False
    return True

def setup_wandb():
    """Setup Weights & Biases with API key."""
    if not check_wandb_api_key():
        logger.warning("Weights & Biases integration will be disabled.")
        return False
    
    try:
        # Set the API key
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        logger.info("Successfully logged in to Weights & Biases")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {str(e)}")
        return False

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
    """Setup the model and tokenizer with qLoRA configuration."""
    logger.info(f"Setting up model and tokenizer for {model_name}")
    
    # Configure 4-bit quantization
    logger.info("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    logger.info("Loading model from Hugging Face Hub...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    logger.info("Configuring LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    logger.info(f"Trainable parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}% of {total_params:,} total)")
    
    return model, tokenizer

def create_training_args(output_dir: str):
    """Create training arguments for GRPO."""
    logger.info("Creating training arguments...")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        max_prompt_length=512,
    )
    logger.info(f"Training configuration:\n{config}")
    return config

def prepare_dataset():
    """Prepare the training and validation datasets."""
    logger.info("Loading TLDR dataset...")
    dataset = load_dataset("trl-lib/tldr")
    logger.info(f"Dataset splits loaded:")
    for split in dataset:
        logger.info(f"- {split}: {len(dataset[split]):,} examples")
    return {
        "train": dataset["train"],
        "validation": dataset["validation"]
    }

def reward_len(completions, **kwargs):
    """Reward function that rewards completions close to 20 characters."""
    rewards = [-abs(20 - len(completion)) for completion in completions]
    if kwargs.get('verbose', False):
        logger.debug(f"Rewards calculated: {rewards}")
    return rewards

def train_model(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    output_dir: str = "grpo_trained_model",
    num_prompts: int = 5
):
    """Main training function."""
    logger.info(f"Starting training process with model: {model_name}")
    
    # Initialize wandb
    run_name = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logger.info(f"Initializing Weights & Biases run: {run_name}")
    
    # Check if wandb is available
    if not setup_wandb():
        logger.warning("Training will proceed without Weights & Biases logging")
        wandb_enabled = False
    else:
        wandb_enabled = True
        wandb.init(project="grpo-training", name=run_name)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Create training arguments
    training_args = create_training_args(output_dir)
    
    # Modify training arguments if wandb is not available
    if not wandb_enabled:
        training_args.report_to = ["none"]
        logger.info("Disabled Weights & Biases reporting in training arguments")
    
    # Prepare dataset
    logger.info("Preparing datasets...")
    datasets = prepare_dataset()
    
    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        reward_funcs=reward_len,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    # Save the model
    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved successfully!")

    # Close wandb if it was initialized
    if wandb_enabled:
        wandb.finish()
        logger.info("Weights & Biases run completed")

    # Test the model before and after training
    test_prompts = [
        "Summarize the benefits of exercise in one sentence.",
        "What is the main idea of machine learning?",
        "Explain quantum computing briefly.",
        "What are the key features of artificial intelligence?",
        "Describe the process of photosynthesis in one line."
    ][:num_prompts]

    logger.info("Starting model evaluation...")
    # Save test results
    results = {
        "before_training": {},
        "after_training": {}
    }

    # Test before training
    logger.info("\n" + "="*50 + "\nTesting model before training:")
    for prompt in test_prompts:
        logger.info(f"\nProcessing prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        results["before_training"][prompt] = response
        logger.info(f"Response: {response}")
        logger.info(f"Response length: {len(response)} characters")

    # Test after training
    logger.info("\n" + "="*50 + "\nTesting model after training:")
    for prompt in test_prompts:
        logger.info(f"\nProcessing prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        results["after_training"][prompt] = response
        logger.info(f"Response: {response}")
        logger.info(f"Response length: {len(response)} characters")

    # Save results to file
    logger.info("\nSaving test results to test_results.json...")
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved successfully!")

    # Print summary statistics
    logger.info("\n" + "="*50 + "\nSummary Statistics:")
    avg_len_before = sum(len(v) for v in results["before_training"].values()) / len(results["before_training"])
    avg_len_after = sum(len(v) for v in results["after_training"].values()) / len(results["after_training"])
    logger.info(f"Average response length before training: {avg_len_before:.2f} characters")
    logger.info(f"Average response length after training: {avg_len_after:.2f} characters")
    logger.info(f"Target length was: 20 characters")
    logger.info("="*50)

    return results

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""
    logger.debug(f"Generating response for prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logger.debug("Starting generation...")
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug(f"Generated response: {response}")
    return response

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Starting GRPO Training Script")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("="*50)
    
    try:
        results = train_model(
            model_name="Qwen/Qwen2-0.5B-Instruct",
            output_dir="grpo_trained_model",
            num_prompts=5
        )
        logger.info("Script completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise 