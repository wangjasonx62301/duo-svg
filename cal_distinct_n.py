import json
from collections import Counter
from transformers import AutoTokenizer

def calculate_distinct_n_with_tokenizer(texts, tokenizer, n):
    if not texts:
        return 0.0

    ngrams = []

    special_ids = set(tokenizer.all_special_ids)

    for text in texts:
        # tokenize to token ids (no special tokens)
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        token_ids = [tid for tid in token_ids if tid not in special_ids]

        # skip too-short sequences
        if len(token_ids) < n:
            continue

        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i:i + n])
            ngrams.append(ngram)

    if not ngrams:
        return 0.0

    distinct_ngrams = set(ngrams)

    if n == 1:
        # get 10 most common unigrams
        most_common_unigrams = Counter(ngrams).most_common(10)  
        print("Most common unigrams (token ids and counts):")
        for unigram, count in most_common_unigrams:
            print(f"Token IDs: {unigram}, Count: {count}")

    return len(distinct_ngrams) / len(ngrams)


def process_json_file(
    file_path,
    tokenizer_name_or_path
):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True
    )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated_seqs = data.get("generated_seqs", [])[:150]
    if not generated_seqs:
        print("No generated sequences found.")
        return

    for n in [1, 2, 3]:
        score = calculate_distinct_n_with_tokenizer(
            generated_seqs, tokenizer, n
        )
        print(f"Distinct-{n}: {score:.4f}")


if __name__ == "__main__":
    file_path = "/home/jasonx62301/for_python/duo-svg/duo-svg/outputs/lm1b/duo_svg_1024-30000/256/samples.json"
    tokenizer_name_or_path = "gpt2"  # or any other pretrained tokenizer
    process_json_file(file_path, tokenizer_name_or_path=tokenizer_name_or_path)
    
    
