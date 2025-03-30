# GRPO Training with qLoRA

This project implements Generative Reward-Penalty Optimization (GRPO) training with qLoRA compression for language models. It uses the Microsoft Phi-2 model as the base model and applies efficient fine-tuning techniques to improve its performance.

## Features

- GRPO training implementation using Hugging Face's TRL library
- qLoRA compression for efficient fine-tuning
- 4-bit quantization for reduced memory usage
- Automatic testing before and after training
- Wandb integration for training monitoring
- Configurable number of test prompts

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Wandb (optional but recommended):
```bash
wandb login
```

## Usage

1. Run the training script:
```bash
python train_grpo.py
```

The script will:
- Load the base model (Phi-2)
- Apply qLoRA compression
- Train the model using GRPO
- Save the trained model
- Test the model before and after training
- Save test results to `test_results.json`

## Model Architecture

- Base Model: Microsoft Phi-2
- Training Method: GRPO (Generative Reward-Penalty Optimization)
- Compression: qLoRA with 4-bit quantization
- LoRA Configuration:
  - r: 16
  - alpha: 32
  - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - dropout: 0.05

## Training Configuration

- Number of epochs: 3
- Batch size: 4
- Gradient accumulation steps: 4
- Learning rate: 2e-4
- Mixed precision training (fp16)
- Maximum prompt length: 512
- Maximum sequence length: 1024

## Example Outputs

### Before Training

```
Prompt: What is machine learning?
Response: [Base model response will be shown here]

Prompt: Explain quantum computing.
Response: [Base model response will be shown here]

[Additional prompts and responses...]
```

### After Training

```
Prompt: What is machine learning?
Response: [Trained model response will be shown here]

Prompt: Explain quantum computing.
Response: [Trained model response will be shown here]

[Additional prompts and responses...]
```

## Model Size and Performance

- Original Phi-2 model size: ~2.7GB
- Compressed model size (with qLoRA): ~1.5GB
- Training memory requirements: ~8GB GPU RAM
- Inference memory requirements: ~4GB GPU RAM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 