from funasr import AutoModel
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
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

# 初始化 SenseVoice 模型
model_name = "iic/SenseVoiceSmall"
asr_model = AutoModel(model=model_name, disable_update=True, device="mps") # change to your own device if necessary

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

@app.get("/status")
async def status():
    return Response(status_code=HTTP_204_NO_CONTENT)
