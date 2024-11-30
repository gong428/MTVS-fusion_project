from fastapi import APIRouter,Form, Depends
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.document_loaders import 
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from app.database.database import get_db,DiscussionTopic
from sqlalchemy.orm import Session
from app.router.tts import text_to_speak

import re
topic_router = APIRouter(prefix="/discussion_topic")

#임베딩 저장 장소
#TEST_DB_PATH = "./emb_db/운수좋은날"
#TEST_DB_PATH = "./emb_db/이솝우화"
TEST_DB_PATH = "./emb_db/백설공주"
#TEST_DB_PATH = "./emb_db/신데렐라"

# 임베딩 도구 세팅
emb = OpenAIEmbeddings()
# 모델 설정
model = ChatOpenAI(model_name="gpt-4o-2024-08-06")
#model = ChatOpenAI(model_name="gpt-4o-mini")



# 트리거 저장소
prompt_dic = {}
# 대화 내용 저장소
memory_store = {}
#model = ChatOllama(model="llama3.1:latest")

# 주제 추천용
system_prompt = """
    너는 4명에 토론자가 독서 토론을 찬반 토론 형식으로 할 수 있는 주제를 추천해주는 어시스턴트야.
    아래에 지침에 따라서 토론주제를 추천해달라는 Question에 찬반 토론 주제를 답변해줘.

    1. 주제 생성은 찬성하는 의견과 반대하는 의견이 명확히 갈릴 수 있는 주제로 생성해.
    2. 설명은 찬성 측과 반대 측으로 나누어 토론을 유도할 수 있게 생성해.
    3. 토론 주제는 Context에 내용에서만 기반해서 주제를 생성해.
    4. 토론 주제는 하나의 주제만 생성해.
    5. 모든 답변은 높임말로 답변해.
    6. 토론 주제 설명은 5줄 이내로 생성해.
    7. 답변은 반드시 (주제 : , 설명 :)방식으로 통일해서 답변해.
    8. 이전에 나온 주제와 비슷하거나 똑같은 주제는 생성하지마.
    9. 설명에서는 예시를 포함하지 말고 간략하게 찬성측과 반대측에서 토론할 내용을 설명해.
    10. 주제의 난이도는 10세에서 13세 나이대가 이해할 수 있는 주제를 생성해.
    11. 찬성측 설명과 반대측 설명 사이에 '|||'을 반드시 포함해서 답변해.
 
    Question : {text} 
    Context : {context} 

    Answer:
"""



#각각의 문서 내용을 줄바꿈으로 구분된 하나의 문자열로 반환하는 역할을 합니다.
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


## 책내용 기반 자유 토론 주제 생성 ai 만들기
@topic_router.post("/topic_suggest_text",tags=['topic_suggest'])
def topic_suggestion(chat_room_id : str = Form(...),db : Session = Depends(get_db)):

    text = '찬반 토론 주제 추천해줘'

    # 대화 기록 저장소 생성
    if chat_room_id not in memory_store:
        memory_store[chat_room_id] = ConversationBufferMemory()

    memory = memory_store[chat_room_id]

    # 임베딩 db 불러오기
    openai_db = Chroma(
        persist_directory=TEST_DB_PATH,
        embedding_function=emb
    )

    #임베딩 db 를 참조한 chattingContents(유저 입력) 유사한 내용 5개까지 찾기
    openai_retriever = openai_db.as_retriever(
        search_kwargs={"k":3}
    )

    #프롬프트에 삽입할 history 형태 만들어주기
    conversation_history = memory.load_memory_variables({})

    # 시스템 프롬프트에서 참조할 context 
    context = openai_retriever.invoke(text)
    #
    custom_prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # 필요한 모든 변수를 템플릿에 전달(질문용 템플릿 작성)
    formatted_custom_prompt = custom_prompt.format(
        history=conversation_history,
        context=context,
        text=text
    )

    rag_chain = (
        {"context":openai_retriever,"text":RunnablePassthrough()}
         | custom_prompt
         | model
         | StrOutputParser()
    )

    response = rag_chain.invoke(formatted_custom_prompt)

    print(response)

    parts = re.split(r'설명\s*:', response)

    # '주제: ' 또는 '주제 :'에 대해 모두 대응하기 위해 정규표현식 사용
    split_result = re.split(r'주제\s*:', parts[0])
    if len(split_result) > 1:
        response_topic = split_result[1].strip()
        # 설명 부분이 없는 경우 처리
        if len(parts) > 1:
            response_content = parts[1].strip()
        else:
            response_content = ''
    else:
        # 원문 형식이 주제: 설명: 형식이 아닐 때
        response_topic = response.strip()  # 원문 전체를 response_topic에 저장
        response_content = ''  # response_content를 빈 문자열로 설정

    #response_topic = parts[0].split('주제:')[1].strip()
    #response_content  = parts[1].strip()

    # 현재 대화와 AI 응답을 메모리에 추가
    memory.save_context({"user": text}, {"ai": response})

    # db에서 room_id가 일치하는 row에서 count 값이 가장 큰 것 찾기
    existing_topic = db.query(DiscussionTopic).filter(DiscussionTopic.room_id == chat_room_id).order_by(DiscussionTopic.count.desc()).first()

    if existing_topic and existing_topic.count is not None:
        count = existing_topic.count + 1
    else :
        count = 1

    db_topic = DiscussionTopic(room_id =chat_room_id,topic = response_topic, decription = response_content,count=count)
    db.add(db_topic)
    db.commit()

    return {'topic':response_topic, 'content':response_content}


'''
## 책내용 기반 자유 토론 주제 생성 ai 만들기
@topic_router.post("/topic_suggest_text",tags=['topic_suggest'])
def topic_suggestion_text(chat_room_id : str = Form(...),db : Session = Depends(get_db)):
    recente_db= db.query(DiscussionTopic).filter(DiscussionTopic.room_id == chat_room_id).order_by(DiscussionTopic.count.desc()).first() 
    topic = recente_db.topic
    content = recente_db.decription
    
    return {'topic':topic, 'content':content}'''