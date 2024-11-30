from fastapi import FastAPI,Form,UploadFile,File,Request,APIRouter
from fastapi.responses import JSONResponse,HTMLResponse,PlainTextResponse,FileResponse,StreamingResponse
from pydantic import BaseModel
import os
from gtts import gTTS
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import io
import numpy as np
from app.router.text_file import text_router
from app.router.tts import tts_router
from app.router.embedding import emb_router
from app.router.topic import topic_router
from app.router.stt import stt_router
from app.router.summary import summary_router

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# 음성을 텍스트로 변환하는 함수
def speak_to_text(audio_content):
    # 여기서 pipe 함수는 음성을 텍스트로 변환하는 가정된 함수입니다.
    result = pipe(audio_content)
    return result["text"]


# 텍스트를 음성으로 변환하는 함수
def text_to_speak(text):
    # Google Text-to-Speech를 사용하여 텍스트를 음성으로 변환
    tts = gTTS(text=text, lang="ko")
    output_buffer = io.BytesIO()
    tts.write_to_fp(output_buffer)
    output_buffer.seek(0)  # 스트림의 시작으로 이동
    return output_buffer

app = FastAPI()
user_dic = {}

UPLOAD_DIR = "upload_audio"  # 음성 파일이 저장된 디렉토리

if not os.path.exists(UPLOAD_DIR) : #아래로 생략할 수 있음
    os.mkdir(UPLOAD_DIR)

@app.get("/")
def root_index():
    return {"messages":"서버 가동 중."}

'''
# 음성 데이터를 받아서 새로운 음성 데이터 반환하기
@app.post("/audio2audio")
async def stream_audio_file(file:UploadFile = File(...), user_id:int = Form(...)):
    user_id += 10
    temp = "upload_audio/temp_audio.wav"

    file_content = await file.read()

    # 파일을 임시 wav 파일로 저장
    with open(temp, "wb") as f:
        f.write(file_content)
    result = pipe(temp)
    text = result["text"]
    text = "흠 이건 티티에스가 잘 적용되었는지 확인하기 위한 코드입니다. " + text
    # 텍스트를 음성으로 변환
    tts = gTTS(text=text, lang="ko")
    output_file_path = "output_audio.mp3"
    tts.save(output_file_path)

    # 저장된 파일을 읽어서 클라이언트에게 스트리밍
    def iterfile():
        with open(output_file_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="audio/mpeg")
'''
'''
# 음성 데이터를 받아서 새로운 음성 데이터 반환하기- 한번에 받아서 한번에 보내기
@app.post("/speak2speak")
async def stream_audio_file(request:Request):
    # 요청 바디를 바이너리로 읽기
    file_content = await request.body()

    # 음성을 텍스트로 변환
    text = speak_to_text(file_content)
    text = "흠 이건 티티에스가 잘 적용되었는지 확인하기 위한 코드입니다. " + text
    
    
    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer = text_to_speak(text)

    # 저장된 파일을 읽어서 클라이언트에게 스트리밍
    def iterfile():
        yield from output_buffer

    return StreamingResponse(iterfile(), media_type="audio/mpeg")


def save_request_data(request: Request, body_data: bytes):
    """
    요청 헤더와 바디 데이터를 파일로 저장하는 함수.
    """
    # 요청의 헤더를 파일에 저장
    with open("request_headers.txt", "w") as header_file:
        headers = dict(request.headers)
        header_content = "\n".join([f"{key}: {value}" for key, value in headers.items()])
        header_file.write(header_content)

    # 요청의 바디를 파일에 저장
    with open("request_body.bin", "wb") as body_file:
        body_file.write(body_data)
'''
'''
# 음성 데이터를 받아서 새로운 음성 데이터 반환하기- 한번에 받아서 한번에 보내기
@app.post("/speak2speak_chunk")
async def stream_audio_file_chunk(request:Request):
    # 모든 청크 데이터를 수신하여 결합하기 위해 빈 BytesIO 버퍼를 생성
    audio_buffer = io.BytesIO()

    # 청크 데이터를 읽어서 버퍼에 추가
    async for chunk in request.stream():
        audio_buffer.write(chunk)

    # 버퍼의 시작으로 이동하여 전체 데이터를 파일처럼 사용 가능하게 설정
    audio_buffer.seek(0)

    # 원본 요청 데이터를 파일로 저장 (추가된 기능)
    #save_request_data(request, audio_buffer.getvalue())

    # 음성을 텍스트로 변환
    text = speak_to_text(audio_buffer.getvalue())
    print("원본 text: ",text)
    text = "이건 티티에스가 잘 적용되었는지 확인하기 위한 코드입니다. " + text
    
    
    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer = text_to_speak(text)

    # 저장된 파일을 읽어서 클라이언트에게 청크 단위로 스트리밍
    def iterfile():
        # 청크 크기를 정의 (예: 1024 바이트)
        chunk_size = 1024
        while True:
            # 청크 단위로 데이터를 읽어 들임
            chunk = output_buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(iterfile(), media_type="audio/mpeg")
'''

# 음성 데이터를 받아서 새로운 음성 데이터 반환하기- 한번에 받아서 한번에 보내기
@app.post("/speak2speak_multipart",tags=["test"])
async def stream_audio_file_multipart(audio_file : UploadFile = File(...)):
    # 모든 청크 데이터를 수신하여 결합하기 위해 빈 BytesIO 버퍼를 생성
    audio_data = await audio_file.read()

    # 음성을 텍스트로 변환
    text = speak_to_text(audio_data)
    print("원본 text: ",text)
    text = "이건 티티에스가 잘 적용되었는지 확인하기 위한 코드입니다. " + text
    
    
    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer = text_to_speak(text)

    # 저장된 파일을 읽어서 클라이언트에게 청크 단위로 스트리밍
    def iterfile():
        # 청크 크기를 정의 (예: 1024 바이트)
        chunk_size = 1024
        while True:
            # 청크 단위로 데이터를 읽어 들임
            chunk = output_buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(iterfile(), media_type="audio/wav")
#ngrok http http://localhost:8080 실행 코드

app.include_router(tts_router)
app.include_router(emb_router)
app.include_router(topic_router)
app.include_router(stt_router)
app.include_router(summary_router)