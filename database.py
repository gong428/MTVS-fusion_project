
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey,DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker
from datetime import datetime
Base = declarative_base()

# 1. 데이터베이스 모델의 스키마 정의
class DiscussionMessage(Base):
    __tablename__ = "discussion_messages" # 테이블 이름
    id = Column(Integer, primary_key = True, autoincrement = True)
    room_id = Column(Integer, index = True)
    user_id = Column(Integer, index = True)
    content = Column(String)
    time = Column(DateTime,default=datetime.utcnow)

class DiscussionTopic(Base):
    __tablename__ = "discussion_topic" # 대화내용
    id = Column(Integer, primary_key=True)
    room_id = Column(Integer, index = True)
    count = Column(Integer, index = True)
    topic = Column(String)
    decription = Column(String)

# 2. 챗봇 데이터를 저장할 스키마 정의



# 2. 데이터베이스 연결
SQLALCHEMY_DATABASE_URL = "sqlite:///./cb_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
# 3. 데이터 베이스 및 테이블 생성
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db # 어떤 요청을 처리할 때만 데이터베이스 세션을 유지
    except:
        db.close() # 요청이 끝나면 데이터베이스 세션을 종료

# -------------------------------------------------------------------

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db # 어떤 요청을 처리할 때만 데이터베이스 세션을 유지
    finally:
        db.close() # 요청이 끝나면 데이터베이스 세션을 종료

# ai 와 소통할 유저 가입시키기
@app.post("/users/")
def create_user(room_id : int, user_id : int, content : str,db : Session = Depends(get_db)):
    db_message = DiscussionMessage(room_id = room_id ,  user_id =  user_id, content = content)
    db.add(db_message)
    db.commit()
    return {"room_id": room_id, "user_id":user_id,"content":content}

# 데이터 확인
@app.get("/users/")
def read_users(db : Session = Depends(get_db)):
    users = db.query(DiscussionMessage).all()
    return users

# 데이터 변경
@app.put("/users/{user_id}")
def update_users(room_id : int, user_id: int,  content : str, db : Session = Depends(get_db)):
    
    db_message = db.query(DiscussionMessage).filter(DiscussionMessage.id == room_id).first()
    if db_message is None:
        print("방이 없어요")
    db_message.room_id = room_id
    db_message.user_id = user_id
    db_message.content = content
    db.commit()
    db.refresh(db_message)

    return {"room_id": room_id, "user_id":user_id, "content":content}

@app.delete("/users/{user_id}")
def delete_users(user_id: int, db : Session = Depends(get_db)):
    db_user = db.query(DiscussionMessage).filter(DiscussionMessage.id == user_id).first()
    db.delete(db_user)
    db.commit()
    return {"user_id": user_id}
'''
@app.post("/chatbot/{user_id}")
def chatbot_conversation(input_text :str, user_id : int, db : Session = Depends(get_db)):
    output_text = "안녕하세요 ai 답변입니다" # ai 답변으로 바꾸기

    user = db.query(DiscussionMessage).filter(DiscussionMessage.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="유저가 없어요")
    
    db_chatbot = Chatbot(input_text=input_text, output_text = output_text, user_id = user_id)
    db.add(db_chatbot)
    db.commit()
    return {"input_text": input_text, "output_text": output_text}'''