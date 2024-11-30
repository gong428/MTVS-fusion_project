from sqlalchemy import Column, Integer, String, create_engine, ForeignKey,DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import sessionmaker
from datetime import datetime
Base = declarative_base()

# 1. 데이터베이스 모델의 스키마 정의
class DiscussionMessage(Base):
    __tablename__ = "discussion_messages" # 테이블 이름
    id = Column(Integer, primary_key = True, autoincrement = True)
    room_id = Column(String)
    user_id = Column(String)
    content = Column(String)
    time = Column(DateTime,default=datetime.utcnow)

class DiscussionTopic(Base):
    __tablename__ = "discussion_topic" # 대화내용
    id = Column(Integer, primary_key=True,autoincrement = True)
    room_id = Column(String)
    count = Column(String)
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
        yield db
    except Exception as e:
        print(f"Database error: {e}")
        raise
    finally:
        db.close()
