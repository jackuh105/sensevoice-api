import io
import torch
import base64
import soundfile as sf
from funasr import AutoModel
from pydub import AudioSegment
from pydantic import BaseModel
from kokoro import KModel, KPipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Body
from starlette.status import HTTP_415_UNSUPPORTED_MEDIA_TYPE, HTTP_422_UNPROCESSABLE_ENTITY, HTTP_204_NO_CONTENT

app = FastAPI()

# CORS Setting
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def en_callable(text):
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    return next(en_pipeline(text)).phonemes

# 初始化 SenseVoice 模型
model_name = "iic/SenseVoiceSmall"
asr_model = AutoModel(model=model_name, disable_update=True, device="mps") # change to your own device if necessary
# 初始化 Kokoro TTS 模型
tts_repo_id = "hexgrad/Kokoro-82M-v1.1-zh"
tts_device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = KModel(repo_id=tts_repo_id).to(tts_device).eval()
tts_pipelines = {
    "en": KPipeline(lang_code="a", repo_id=tts_repo_id, model=tts_model),
    "zh": KPipeline(lang_code="z", repo_id=tts_repo_id, model=tts_model, en_callable=en_callable)
}

# Pydantic 模型定義輸入結構
class TTSRequest(BaseModel):
    text: str
    language_code: str = "zh"
    voice_settings: str = "zf_001"

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    # 驗證音頻文件是否存在
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # 驗證音頻格式
    file_extension = audio_file.filename.split('.')[-1].lower()
    if file_extension not in ["wav", "mp3", "opus", "webm"]:
        raise HTTPException(
            status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio format. Only wav, mp3, opus, and webm are allowed."
        )

    # 讀取音頻並驗證採樣率
    audio = AudioSegment.from_file(audio_file.file, format=file_extension)
    sample_rate = audio.frame_rate
    if sample_rate < 16000:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Sample rate must be at least 16000 Hz"
        )
    if sample_rate > 16000:
        audio = audio.set_frame_rate(16000)

    # 轉錄音頻
    try:
        res = asr_model.generate(input=audio.raw_data, language="auto", use_itn=True)
        transcription = res[0]["text"].split("|>")[-1].strip()
        return {"status": "success", "transcription": transcription or "No speech detected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest = Body(...)):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    # 驗證 language_code
    if request.language_code not in ["en", "zh"]:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported language code. Only 'en' and 'zh' are allowed."
        )

    try:
        # 根據語言選擇對應的 pipeline
        pipeline = tts_pipelines[request.language_code]
        # 生成音頻
        generator = pipeline(request.text, voice=request.voice_settings)
        result = next(generator)
        audio_data = result.audio

        # 將音頻數據轉為字節流
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, samplerate=24000, format="WAV")
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

        # 將字節數據編碼為 Base64 字符串
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {"status": "success", "audio_data": audio_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.get("/status")
async def status():
    return Response(status_code=HTTP_204_NO_CONTENT)
