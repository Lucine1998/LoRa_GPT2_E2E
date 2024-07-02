import json

def format_convert(read_file, write_file):
    with open(read_file, "r", encoding="utf8") as reader,          open(write_file, "w", encoding="utf8") as writer:
        for line in reader:
            items = line.strip().split("||")
            context = items[0]
            completion = items[1].strip("\n")
            x = {"context": context, "completion": completion}
            writer.write(json.dumps(x) + "\n")

def preprocess_data(data_dir="data"):
    format_convert(os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "train_formatted.jsonl"))
    format_convert(os.path.join(data_dir, "test.txt"), os.path.join(data_dir, "test_formatted.jsonl"))
