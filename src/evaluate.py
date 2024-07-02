from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import GPT2Tokenizer
from datasets import load_metric

def generate_text(model, tokenizer, input_text, max_length=30):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def evaluate_model(model, dataloader, tokenizer):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for inputs, labels, masks in tqdm(dataloader, desc="Evaluating"):
            outputs = model.generate(inputs, max_length=50)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(refs)

    rouge = load_metric("rouge")
    bleu = load_metric("bleu")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    bleu_result = bleu.compute(predictions=predictions, references=references)
    return rouge_result, bleu_result
