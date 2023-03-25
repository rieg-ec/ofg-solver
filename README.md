## Docs

### `embeddings.py`

used to create embeddings of documents.

### `document_summarization.py`

used to create summaries of large documents. Mostly
to provide summaries to indexes through their `set_text` method.

Could also be useful just to generate summaries:

`python3 document_summarization.py path_to_document.txt --max-tokens=1200 `

### `pdf_to_text.py`

used to convert pdfs to plain text files. It runs each paragraph
through Openai's gpt-3.5-turbo model to fix formatting and grammatical issues, so beware it can get expensive quickly if
the document is _too_ lagre.

### `transcript.py`

used to extract transcripts from audio files using Openai's whisper model.

### `query.py` (WIP)

used to query indexes.

The indexes are built using the `embeddings.py` script.

all indexes that match `./indexes/index_*.json` will be used.
