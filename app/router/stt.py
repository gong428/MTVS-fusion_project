from fastapi import APIRouter, Depends,HTTPException, status,UploadFile,File,Form
from fastapi import Response
from sqlalchemy.orm import Session
from app.database.database import get_db, DiscussionMessage
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

stt_router = APIRouter(prefix='/stt')

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


# 토론 내용을 저장하기 위한 코드
@stt_router.post("/input_discussion_content",tags=["stt"])
async def input_discussion_content(chat_room_id : str = Form(...),user_id : str = Form(...), file : UploadFile = File(...),db : Session = Depends(get_db)):
   # 모든 청크 데이터를 수신하여 결합하기 위해 빈 BytesIO 버퍼를 생성
    audio_data = await file.read()

    # 음성을 텍스트로 변환
    text = speak_to_text(audio_data)
    print("원본 text: ",text)
    db_message = DiscussionMessage(room_id =  chat_room_id ,  user_id =  user_id, content = text)
    db.add(db_message)
    db.commit()

    # 폼 데이터 형식으로 응답을 반환
    form_data = f"room_id={chat_room_id}&user_id={user_id}&text={text}"
    return Response(content=form_data, media_type="application/x-www-form-urlencoded")

# 데이터 확인
@stt_router.get("/db_test/",tags=["stt"])
def read_db(db : Session = Depends(get_db)):
    users = db.query(DiscussionMessage).all()
    return users

@stt_router.delete("/db_test/{room_id}",tags=["stt"])
def delete_db(chat_room_id: str, db : Session = Depends(get_db)):
    try:
        # Delete all rows where DiscussionMessage.room_id matches the input room_id
        deleted_count = db.query(DiscussionMessage).filter(DiscussionMessage.room_id == chat_room_id).delete()
        
        # Commit the changes to the database
        db.commit()
        
        # Check if any row was deleted
        if deleted_count == 0:
            return {"message": "No records found with the given room_id.", "room_id":chat_room_id}
        
        # Return the room_id that was used to delete the rows
        return {"message": "Records deleted successfully.", "room_id":chat_room_id}

    except Exception as e:
        # Log the exception and return an error response
        return {"error": str(e)}

## 음성 파일 받아서 텍스트로 변환하고 db 저장 - user별 토론 내용 저장 기능

## 