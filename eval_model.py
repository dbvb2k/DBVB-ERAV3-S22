import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import logging
import json
import sys
from typing import Dict, List, Tuple
import numpy as np

NUM_DASHES = 200

# Configure logging with both file and console handlers
def setup_logging(output_dir: str):
    """Setup logging configuration with both file and console handlers."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = os.path.join(output_dir, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler - use utf-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with error handling for character encoding
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                msg = self.format(record)
                # Replace problematic characters with '?'
                safe_msg = msg.encode('cp1252', errors='replace').decode('cp1252')
                self.stream.write(safe_msg + self.terminator)
                self.stream.flush()
    
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging("logs")

def load_model_and_tokenizer(model_path: str, base_model: str = "facebook/opt-350m"):
    """Load a trained model and its tokenizer."""
    logger.info("\n" + "="*NUM_DASHES)
    logger.info(f"Loading model and tokenizer from {model_path}")
    logger.info("-"*NUM_DASHES)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Calculate memory limits (90% for model, 10% for buffer)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        model_memory = int(0.9 * total_memory)
        buffer_memory = int(0.1 * total_memory)
        max_memory = {0: model_memory}
        logger.info(f"Total GPU memory: {total_memory/1024**3:.2f} GB")
        logger.info(f"Model memory limit: {model_memory/1024**3:.2f} GB")
        logger.info(f"Buffer memory: {buffer_memory/1024**3:.2f} GB")
    else:
        max_memory = None
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set padding token to EOS token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        max_memory=max_memory,
    )
    
    # Load LoRA weights with OPT-specific configuration
    logger.info("Loading LoRA weights...")
    try:
        # First try loading with default configuration
        model = PeftModel.from_pretrained(base_model, model_path)
    except ValueError as e:
        if "Target modules" in str(e):
            logger.info("Retrying with OPT-specific target modules...")
            # Create a new LoRA config with OPT-specific target modules
            from peft import LoraConfig
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],  # OPT-specific attention modules
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # Load the model with the new configuration
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                config=lora_config
            )
        else:
            raise e
    
    # Print model statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\nModel Statistics:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model with improved parameters for accuracy."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Enhanced generation parameters for more accurate responses
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.4,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        min_new_tokens=30,
        no_repeat_ngram_size=3,
        top_p=0.80,
        top_k=40,
        repetition_penalty=1.5,
        length_penalty=1.0,
        early_stopping=True,
    )
    
    # Get the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (remove the prompt)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    response = generated_text[len(prompt_text):].strip()
    
    # Clean up the response
    response = ' '.join(response.split())
    response = ''.join(char for char in response if ord(char) < 128)
    
    # Post-process the response
    if response:
        # Ensure proper capitalization
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
        
        # Ensure it ends with appropriate punctuation
        if response[-1] not in ['.', '!', '?']:
            response += '.'
            
        # Trim to one sentence if it's too long
        first_sentence_end = next((i for i, char in enumerate(response) if char in ['.', '!', '?']), len(response))
        if first_sentence_end < len(response) - 1 and first_sentence_end > 40:
            response = response[:first_sentence_end + 1]
    
    return response

def reward_relevance_and_accuracy(completions, prompts=None, **kwargs):
    """Enhanced reward function that rewards relevance, accuracy and appropriate format."""
    rewards = []
    
    # Common words that don't indicate relevance
    common_words = {'the', 'a', 'an', 'in', 'of', 'to', 'and', 'or', 'is', 'are', 'what', 
                   'how', 'why', 'when', 'one', 'you', 'your', 'my', 'i', 'we', 'it', 
                   'that', 'this', 'these', 'those', 'be', 'been', 'being', 'was', 'were',
                   'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must',
                   'have', 'has', 'had', 'do', 'does', 'did', 'am', 'about', 'as', 'at',
                   'by', 'for', 'from', 'on', 'with'}
    
    # Domain-specific important keywords for each domain
    domain_keywords = {
        'exercise': {'exercise', 'physical', 'health', 'fitness', 'cardiovascular', 'strength', 'endurance', 'muscle', 'weight', 'metabolism', 'mental'},
        'machine learning': {'machine', 'learning', 'algorithm', 'data', 'model', 'training', 'prediction', 'pattern', 'neural', 'classification', 'regression'},
        'quantum computing': {'quantum', 'computing', 'qubits', 'superposition', 'entanglement', 'algorithm', 'bits', 'computation', 'mechanics'},
        'artificial intelligence': {'ai', 'artificial', 'intelligence', 'learning', 'neural', 'network', 'algorithm', 'reasoning', 'cognitive', 'autonomous'},
        'photosynthesis': {'photosynthesis', 'plants', 'chlorophyll', 'light', 'energy', 'carbon', 'dioxide', 'glucose', 'oxygen', 'sunlight', 'water'},
    }
    
    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        # 1. Length component - optimal range 50-150 chars
        length = len(completion)
        if length < 50:
            length_score = -0.5 * (50 - length) / 50
        elif length > 150:
            length_score = -0.3 * (length - 150) / 150
        else:
            length_score = 0.5
        
        # 2. Extract the actual question topic from the prompt
        prompt_topic = prompt.lower()
        topic_category = None
        
        # Determine which domain this prompt belongs to
        for domain in domain_keywords:
            if domain in prompt_topic:
                topic_category = domain
                break
        
        # Extract the topic from "summarize X", "describe Y" formats
        for prefix in ["summarize ", "describe ", "explain ", "what is ", "what are "]:
            if prefix in prompt_topic:
                prompt_topic = prompt_topic[prompt_topic.find(prefix) + len(prefix):]
                break
                
        # Remove common trailing phrases
        for suffix in [" in one sentence", " in one line", " briefly", " in a few words"]:
            if suffix in prompt_topic:
                prompt_topic = prompt_topic[:prompt_topic.find(suffix)]
        
        # 3. Calculate keyword matching - basic relevance
        prompt_words = set(prompt_topic.split())
        completion_words = set(completion.lower().split())
        
        # Filter out common words
        prompt_keywords = prompt_words - common_words
        
        # Count matching keywords
        matches = prompt_keywords.intersection(completion_words)
        keyword_match_ratio = len(matches) / (len(prompt_keywords) + 1e-6) if prompt_keywords else 0
        
        # 4. Domain-specific keyword matching - topic relevance
        domain_relevance = 0
        if topic_category:
            domain_matches = domain_keywords[topic_category].intersection(completion_words)
            domain_relevance = len(domain_matches) / len(domain_keywords[topic_category]) * 1.5
        
        # 5. Clarity and conciseness factors
        sentence_count = len([s for s in completion.split('.') if len(s.strip()) > 0])
        clarity_score = 0.3 if sentence_count <= 2 else -0.1 * (sentence_count - 2)
        
        # 6. Format quality
        has_proper_capitalization = completion[0].isupper() if completion else False
        has_ending_punctuation = completion[-1] in ['.', '!', '?'] if completion else False
        format_score = 0.3 * (has_proper_capitalization + has_ending_punctuation) / 2
        
        # 7. Penalize first-person and second-person references
        personal_references = {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself'}
        personal_ref_count = len(personal_references.intersection(completion_words))
        personal_penalty = -0.2 * personal_ref_count if personal_ref_count > 0 else 0
        
        # 8. Calculate final reward with weighted components
        relevance_score = (keyword_match_ratio * 2) + (domain_relevance * 1.5)
        reward = relevance_score + length_score + format_score + clarity_score + personal_penalty
        rewards.append(reward)
    
    return rewards

def evaluate_model(model_path: str, num_prompts: int = 5, num_candidates: int = 5):
    """Evaluate a trained model using multiple prompts and metrics."""
    logger.info("\n" + "="*NUM_DASHES)
    logger.info("Starting Model Evaluation")
    logger.info("-"*NUM_DASHES)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Test prompts
    test_prompts = [
        "Summarize the benefits of exercise in one sentence.",
        "What is the main idea of machine learning?",
        "Explain quantum computing briefly.",
        "What are the key features of artificial intelligence?",
        "Describe the process of photosynthesis in one line."
    ][:num_prompts]
    
    # Initialize results storage
    results = {
        "prompts": test_prompts,
        "responses": [],
        "scores": [],
        "metrics": {}
    }
    
    # Generate and evaluate responses
    logger.info("\nGenerating and evaluating responses...")
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nPrompt {i}/{len(test_prompts)}: {prompt}")
        
        # Generate multiple candidates and pick the best one
        candidates = []
        candidate_scores = []
        
        for _ in range(num_candidates):
            candidate = generate_response(model, tokenizer, prompt)
            candidates.append(candidate)
            score = reward_relevance_and_accuracy([candidate], [prompt])[0]
            candidate_scores.append(score)
        
        # Select the best response
        best_idx = np.argmax(candidate_scores)
        best_response = candidates[best_idx]
        best_score = float(candidate_scores[best_idx])  # Convert to Python float
        
        results["responses"].append(best_response)
        results["scores"].append(best_score)
        
        logger.info(f"Best Response: \"{best_response}\"")
        logger.info(f"Response Length: {len(best_response)} characters")
        logger.info(f"Quality Score: {best_score:.2f}")
    
    # Calculate metrics and convert to Python native types
    results["metrics"] = {
        "average_score": float(np.mean(results["scores"])),
        "std_score": float(np.std(results["scores"])),
        "average_length": float(np.mean([len(r) for r in results["responses"]])),
        "std_length": float(np.std([len(r) for r in results["responses"]])),
        "min_score": float(np.min(results["scores"])),
        "max_score": float(np.max(results["scores"])),
        "min_length": int(np.min([len(r) for r in results["responses"]])),
        "max_length": int(np.max([len(r) for r in results["responses"]]))
    }
    
    # Print metrics summary
    logger.info("\n" + "="*NUM_DASHES)
    logger.info("Evaluation Metrics Summary")
    logger.info("-"*NUM_DASHES)
    logger.info(f"Average Quality Score: {results['metrics']['average_score']:.2f} ± {results['metrics']['std_score']:.2f}")
    logger.info(f"Score Range: {results['metrics']['min_score']:.2f} - {results['metrics']['max_score']:.2f}")
    logger.info(f"Average Response Length: {results['metrics']['average_length']:.1f} ± {results['metrics']['std_length']:.1f} characters")
    logger.info(f"Length Range: {results['metrics']['min_length']} - {results['metrics']['max_length']} characters")
    logger.info("="*NUM_DASHES)
    
    # Save results
    output_dir = os.path.dirname(model_path)
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    return results

if __name__ == "__main__":
    logger.info("\n" + "="*NUM_DASHES)
    logger.info("Starting Model Evaluation Script")
    logger.info("="*NUM_DASHES)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    logger.info("-"*NUM_DASHES + "\n")
    
    try:
        # List available models
        base_dir = "trained_models"
        if not os.path.exists(base_dir):
            logger.error(f"Directory {base_dir} does not exist!")
            sys.exit(1)
            
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not model_dirs:
            logger.error(f"No trained models found in {base_dir}!")
            sys.exit(1)
            
        logger.info("Available trained models:")
        for i, model_dir in enumerate(model_dirs, 1):
            logger.info(f"{i}. {model_dir}")
            
        # Get user input for model selection
        while True:
            try:
                selection = int(input("\nEnter the number of the model to evaluate (or 0 to exit): "))
                if selection == 0:
                    sys.exit(0)
                if 1 <= selection <= len(model_dirs):
                    break
                logger.error("Invalid selection. Please try again.")
            except ValueError:
                logger.error("Please enter a valid number.")
        
        # Evaluate selected model
        model_path = os.path.join(base_dir, model_dirs[selection-1])
        results = evaluate_model(model_path, num_prompts=5, num_candidates=5)
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
        raise 