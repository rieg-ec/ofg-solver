import openai
import os
import re
import tiktoken
import argparse
from dotenv import load_dotenv

"""
Summarize a text document with a specified max tokens limit.

usage: python3 document_summarization.py document.txt --max-tokens=1200

output will go into ./summaries/document.txt
"""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(text))
    return token_count


def split_into_chunks(document, max_tokens=1200, model="gpt-3.5-turbo"):
    sentences = re.split(r"(?<=[.!?]) +", document)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_token_count = count_tokens(sentence, model)

        if current_token_count + sentence_token_count <= max_tokens:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_text(document, filepath=None, model="gpt-3.5-turbo", max_tokens=2000):
    if filepath:
        filename_without_path_or_extension = re.sub(
            r"^.*/([^/]+)\.txt$", r"\1", filepath
        )
        summary_filepath = f"./summaries/{filename_without_path_or_extension}.txt"

        if os.path.exists(summary_filepath):
            print("Summary already exists, skipping summary generation...")
            with open(summary_filepath, "r") as file:
                return file.read()

    messages = [
        {
            "role": "system",
            "content": "You're an AI that summarizes texts in different languages. You receive a text \
                        and return the summary as if it were written by the same author, in the same language, in \
                        first person. Don't try to explain anything, just return less text \
                        than it is given (summary should be roughly half the size of the original text), \
                        but with the same information. Also, it is VERY IMPORTANT to keep the language \
                        it was written on. If it is written in spanish, the output must be also spanish.",
        },
        {"role": "user", "content": f"Summarize this paragraph in 6 lines: {document}"},
    ]

    if count_tokens(document) > max_tokens:
        chunks = split_into_chunks(document, max_tokens, model)

        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = summarize_text(chunk, model=model, max_tokens=max_tokens)
            summarized_chunks.append(summarized_chunk)
        return summarize_text(
            " ".join(summarized_chunks), model=model, max_tokens=max_tokens
        )

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=1,
        temperature=0.5,
    )

    summary = response.choices[0].message["content"].strip()

    print("\n\n-------DOCUMENT-------\n\n", document)
    print("\n\n-------SUMMARY----\n\n", summary, "\n\n")

    if count_tokens(summary) > max_tokens:
        return summarize_text(summary, model=model, max_tokens=max_tokens)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize a text document with a specified max tokens limit."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="file path to the text document to summarize",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens allowed for a chunk. Default is 1200",
    )

    args = parser.parse_args()

    document_path = args.filepath
    with open(document_path, "r") as f:
        summary = summarize_text(
            f.read(), filepath=document_path, max_tokens=args.max_tokens
        )

    filename_without_path_or_extension = re.sub(
        r"^.*/([^/]+)\.txt$", r"\1", document_path
    )

    with open(f"./summaries/{filename_without_path_or_extension}.txt", "w") as f:
        f.write(summary)
