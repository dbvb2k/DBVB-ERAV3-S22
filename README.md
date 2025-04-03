# GRPO Training with OPT-350M and qLoRA

This project implements Gradient Reinforcement Policy Optimization (GRPO) training on Facebook's OPT-350M model using qLoRA for efficient fine-tuning. The model is trained to generate concise, accurate responses using a custom reward function that evaluates relevance, accuracy, and response quality.

## Features

- Uses Facebook's OPT-350M as the base model
- Implements qLoRA for efficient fine-tuning
- Uses GRPO trainer from Hugging Face's TRL library
- Includes before/after training comparisons
- Configurable number of test prompts
- TensorBoard integration for training monitoring
- Enhanced reward function for better response quality
- Memory-efficient training with 4-bit quantization

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python train_grpo.py
```

The script will:
1. Load and prepare the OPT-350M model with qLoRA
2. Test the model before training
3. Train the model using GRPO
4. Test the model after training
5. Save the results to `test_results.json`

## Model Configuration

- Base model: Facebook OPT-350M
- Quantization: 4-bit with qLoRA
- LoRA parameters:
  - r: 16
  - alpha: 32
  - target_modules: ["q_proj", "v_proj"]
  - dropout: 0.05
- Training parameters:
  - Batch size: 2
  - Gradient accumulation steps: 2
  - Learning rate: 2e-5
  - Epochs: 3
  - Max steps: 2000
  - Warmup ratio: 0.1
  - Optimizer: paged_adamw_32bit
  - Learning rate scheduler: cosine

## Reward Function

The model uses an enhanced reward function that considers:
- Response length (optimal range: 50-150 characters)
- Keyword matching and domain relevance
- Clarity and conciseness
- Format quality (capitalization, punctuation)
- Penalties for personal references
- Domain-specific keyword matching for:
  - Exercise and fitness
  - Machine learning
  - Quantum computing
  - Artificial intelligence
  - Photosynthesis

## Results

The model's performance is evaluated before and after training using the same prompts. Results are saved in `test_results.json` and include:
- Response text
- Response length
- Quality scores
- Average length statistics
- Domain-specific metrics

## Training Results

Here are some example responses before and after training:

### Before Training
1. Prompt: "Summarize the benefits of exercise in one sentence."
   Response: "Exercise improves physical health, mental well-being, and overall quality of life."
   Length: 71 characters

2. Prompt: "What is the main idea of machine learning?"
   Response: "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance over time."
   Length: 108 characters

### After Training
1. Prompt: "Summarize the benefits of exercise in one sentence."
   Response: "Regular exercise improves physical health, mental wellbeing, and extends lifespan."
   Length: 65 characters

2. Prompt: "What is the main idea of machine learning?"
   Response: "Machine learning enables computers to improve from experience without explicit programming."
   Length: 72 characters

## Model Evaluation

To evaluate a trained model:
```bash
python eval_model.py
```

This will:
1. List available trained models
2. Allow selection of a model to evaluate
3. Generate responses for test prompts
4. Calculate and display metrics
5. Save evaluation results to `evaluation_results.json`

## Repository

GitHub Repository: [DBVB-ERAV3-S22](https://github.com/dbvb2k/DBVB-ERAV3-S22)

## License

MIT License 