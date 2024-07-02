import torch
from transformers import AutoTokenizer
from src.downloader import download_e2e_data
from src.preprocessor import preprocess_data
from src.dataloader import get_dataloader
from src.model import get_model
from src.train import train, save_model
from src.evaluate import evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    lora_dim = 128
    batch_size = 8
    block_size = 512
    num_epochs = 3

    download_e2e_data()
    preprocess_data()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader = get_dataloader("data/train_formatted.jsonl", tokenizer, batch_size, block_size, device)
    test_dataloader = get_dataloader("data/test_formatted.jsonl", tokenizer, batch_size, block_size, device, shuffle=False)

    model = get_model(model_name, lora_dim, device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    train(model, train_dataloader, optimizer, device, num_epochs)
    save_model(model)

    rouge_result, bleu_result = evaluate_model(model, test_dataloader, tokenizer)
    print("ROUGE:", rouge_result)
    print("BLEU:", bleu_result)

if __name__ == "__main__":
    main()
