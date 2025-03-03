# STT/TTS API Server

## Summary

STT/ TTS API server using `SenseVoice` for STT, `kokoro` v1.1 for TTS and `FastAPI` for the server.
Provides 3 API:
1. `/transcribe`: transcribe the received audio file and return the transcription.
2. `/text-to-speech`: generate audio file from the received text.
3. `/status`: health check.

## Quick Start

1. Create a virtual environment.

```bash
python -m venv venv
```

2. Activate the virtual environment and install the dependencies.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

3. Start the server

```bash
uvicorn main:app --reload
```

## File structure

```
stt-api/
├── main.py         
├── .gitignore      
├── requirements.txt
└── README.md   
```