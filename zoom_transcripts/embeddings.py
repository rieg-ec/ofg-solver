import os
import argparse
import re
from dotenv import load_dotenv
from document_summarization import summarize_text

"""
generates embeddings for a text file and saves them to disk as indexes.

usage: python3 embeddings.py document.txt
"""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from llama_index import GPTSimpleVectorIndex, Document


def build_index(filepath):
    with open(filepath, "r") as file:
        data = file.read()

    document = Document(data)
    index = GPTSimpleVectorIndex([document])

    summary = summarize_text(data, filepath=filepath)
    index.set_text(summary)

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from a text file."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="filepath to the text document.",
    )

    args = parser.parse_args()

    index = build_index(args.filepath)
    filename_without_path_or_extension = re.sub(
        r"^.*/([^/]+)\.txt$", r"\1", args.filepath
    )
    index.save_to_disk(f"indexes/index_{filename_without_path_or_extension}.json")
