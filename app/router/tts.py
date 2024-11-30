from fastapi import APIRouter,Form
from fastapi.responses import StreamingResponse
import io
from gtts import gTTS
from pydub import AudioSegment
# 라우터 선언
tts_router = APIRouter(prefix='/tts')


# 텍스트를 음성으로 변환하는 함수
def text_to_speak(text):
    # Google Text-to-Speech를 사용하여 텍스트를 음성으로 변환
    tts = gTTS(text=text, lang="ko")
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)  # 스트림의 시작으로 이동

    # MP3를 WAV로 변환
    audio = AudioSegment.from_file(mp3_buffer, format="mp3")
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    return wav_buffer

# 음성 파일을 chunk로 쪼개서 보내주기
def iterfile(output_buffer):
        # 청크 크기를 정의 (예: 1024 바이트)
        chunk_size = 1024
        while True:
            # 청크 단위로 데이터를 읽어 들임
            chunk = output_buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk


## 단순히 텍스트를 받아서 음성으로 변환해서 보내기
# 입력 데이터 - 텍스트 데이터 -> 이거 나보고 알아서 만들라고 할 수 있을 것 같음
# 출력 데이터 - 음성 데이터 
@tts_router.post('/tts')
async def guide_voice(text: str = Form(...)):


    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer = text_to_speak(text)
    
    return StreamingResponse(iterfile(output_buffer), media_type="audio/wav")


@tts_router.post('/topic_total_voice')
async def topic_total_voice(topic: str = Form(...),content: str = Form(...)):

    total = topic +'  ' + content

    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer_total = text_to_speak(total)
    
    return StreamingResponse(iterfile(output_buffer_total ), media_type="audio/wav")

@tts_router.post('/topic')
async def topic_voice(topic: str = Form(...)):

    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer_topic = text_to_speak(topic)
    
    return StreamingResponse(iterfile(output_buffer_topic ), media_type="audio/wav")

@tts_router.post('/content')
async def topic_voice(content: str = Form(...)):

    # 텍스트를 음성으로 변환 - 함수로 적용하는게 좋아보임
    output_buffer_content = text_to_speak(content)
    
    return StreamingResponse(iterfile(output_buffer_content), media_type="audio/wav")