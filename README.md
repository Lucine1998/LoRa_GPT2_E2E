# GPT-2 Fine-Tuning with LoRA on E2E Dataset

This repository contains code for fine-tuning GPT-2 using Low-Rank Adaptation (LoRA) on the E2E dataset. The process involves downloading the dataset, preprocessing it, creating data loaders, applying LoRA to the GPT-2 model, training the model, and evaluating it using ROUGE and BLEU metrics.

## Structure

- `data/`: Contains the training and testing data.
- `src/`: Contains the modules for downloading, preprocessing, loading data, defining the model, training, and evaluation.
- `main.py`: The main script to orchestrate the process.
- `requirements.txt`: Python dependencies.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Notes

- Ensure you have a GPU available for training to speed up the process.
- Adjust hyperparameters like `batch_size` and `num_epochs` as needed.
