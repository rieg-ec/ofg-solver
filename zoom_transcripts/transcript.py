import os
import glob
import openai
import argparse
from dotenv import load_dotenv

"""
generates a transcript from chunks of audio files.

audio chunks must be inside ./audios, under the format ./audios/<dirname>/chunk_001.mp3.

output will go into ./transcripts/<dirname>.txt

usage: python3 transcript.py 'chunks_directory'
"""

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio from mp3.")
    parser.add_argument(
        "dirname",
        type=str,
        help="directory with mp3 chunkfiles.",
    )

    args = parser.parse_args()
    chunk_files = sorted(glob.glob(f"./audios/{args.dirname}/chunk_*.mp3"))

    transcripts = []
    for chunk_file in chunk_files:
        print(f"Transcribing {chunk_file}")
        transcript = transcribe_audio(chunk_file)
        transcripts.append(transcript["text"])

        complete_transcript = "\n".join(transcripts)

        with open(f"./transcripts/{args.dirname}.txt", "w") as transcript_file:
            transcript_file.write(complete_transcript)
