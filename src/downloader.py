import os
import requests

def download_file(url, dest):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest, 'wb') as file:
        file.write(response.content)

def download_e2e_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    train_url = "https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/train.txt"
    test_url = "https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/test.txt"
    
    download_file(train_url, os.path.join(data_dir, "train.txt"))
    download_file(test_url, os.path.join(data_dir, "test.txt"))
