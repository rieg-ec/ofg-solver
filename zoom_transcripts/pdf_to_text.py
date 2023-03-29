import argparse
import os
import re
import pdfplumber
import openai
from dotenv import load_dotenv

"""
extracts text from a pdf, running each page through the
gpt-3.5-turbo model to fix formatting issues.

usage: python3 pdf_to_text.py document.txt
"""

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def fix_formatting_issues(text):
    messages = [
        {
            "role": "system",
            "content": "You are an AI that fixes formatting and grammatical errors in text. You are prohibited from \
                        saying anything, just fix the formatting and grammatical errors. These could be missing spaces \
                        between words, encoding issues, bad grammar, etc. You receive some text, and output the fixed text \
                        and nothing else.",
        },
        {"role": "user", "content": f'Fix this paragraph: "{text}"'},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024,
        n=1,
        temperature=0.5,
    )

    return response.choices[0].message["content"].strip()


def extract_text_from_pdf(pdf_file_path, output_file_path, format=True):
    with pdfplumber.open(pdf_file_path) as pdf:
        pages = []

        for page in pdf.pages:
            text = page.extract_text()
            if format:
                text = fix_formatting_issues(text)
            print(text)
            pages.append(text)

        formatted_text = "\n".join(pages)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(formatted_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a pdf.")
    parser.add_argument(
        "filepath",
        type=str,
        help="file path to the pdf file",
    )

    parser.add_argument(
        "--skip-format",
        default=False,
        action="store_true",
        help="whether to pass the document throught chatGPT to fix formatting issues. \
            Helpful when pdf output has a bunch of errors.",
    )

    args = parser.parse_args()

    filename_without_extension = re.sub(r"\.pdf$", "", args.filepath)
    extract_text_from_pdf(
        args.filepath,
        f"{filename_without_extension}.txt",
        format=(not args.skip_format),
    )
