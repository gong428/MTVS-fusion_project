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
from sqlalchemy.orm import Session
from app.database.database import get_db, DiscussionMessage
import json

summary_router = APIRouter(prefix="/discussion_summary")

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
'''(system_prompt = """
    너는 토론 내용을 user_id 별로 요약하고 전체적인 토론 내용을 요약해주고 정리해주는 어시스턴트야.
    아래에 지침에 따라서 토론 내용을 요약해달라는 Question에 Context를 기반으로 답변해줘.
    1. 처음은 user_id별로 토론자들의 내용을 개인별로 요약해줘.
    2. 토론자들 개인별 요약이 끝나면 전체적인 토론 내용을 요약해줘.
    3. 모든 답변은 높임말로 답변해.
    4. 내용 요약은 user_id 별로 3줄 이상 넘어가면 안되.
    5. 'user1의 의견은 ~~이고 user2의 의견은 ~~, 전체적으로 ~~' 이것 처럼 user_id별로 내용 요약을 문장으로 설명해.

    Question : {text} 
    Context : {context} 

    Answer:
""")'''


system_prompt = """
    너는 전체적인 토론 내용을 요약해주고 정리해주는 어시스턴트야.
    아래에 지침에 따라서 토론 내용을 요약해달라는 Question에 Context를 기반으로 답변해줘.
    1. 토론 내용에 어떤 의견이 있었는지 요약해서 답변해줘.
    2. 토론 내용에 주로 논점이 어떤 것인지 알려줘.
    3. 마지막에는 토론자에게 감사와 마무리 멘트를 해줘.
    4. 답변은 정리형태가 아닌 답변형태로 작성해줘.

    Question : {text} 
    Context : {context} 

    Answer:"""


#각각의 문서 내용을 줄바꿈으로 구분된 하나의 문자열로 반환하는 역할을 합니다.
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_message(session: Session, chat_room_id: str):
    # user_id와 content를 딕셔너리 형식으로 변환하여 반환하도록 수정
    messages = session.query(DiscussionMessage.user_id, DiscussionMessage.content).filter(DiscussionMessage.room_id ==chat_room_id).order_by(DiscussionMessage.time).all()
    
    # 객체를 JSON 직렬화 가능한 딕셔너리 리스트로 변환
    return [{"user_id": message.user_id, "content": message.content} for message in messages]

@summary_router.post('/test_load_db',tags=['topic_summary'])
def test_load_db(chat_room_id : str = Form(...),db : Session = Depends(get_db)):
    messages = get_message(db, chat_room_id)
    return messages


## 토론내용 기반 토론 내용 요약 ai
@summary_router.post("/discussion_summary",tags=['topic_summary'])
def discussion_summary(chat_room_id : str = Form(...),db : Session = Depends(get_db)):

    text = '토론 내용 요약해줘'

    # 대화 기록 저장소 생성
    if chat_room_id not in memory_store:
        memory_store[chat_room_id] = ConversationBufferMemory()

    memory = memory_store[chat_room_id]

    #프롬프트에 삽입할 history 형태 만들어주기
    conversation_history = memory.load_memory_variables({})

    #
    custom_prompt = ChatPromptTemplate.from_template(system_prompt)
    
    discussion_contents = get_message(db, chat_room_id)

    # messages를 JSON 형식으로 변환
    context = "\n\n".join([content['content'] for content in discussion_contents])
    print(context)

    # 필요한 모든 변수를 템플릿에 전달(질문용 템플릿 작성)
    formatted_custom_prompt = custom_prompt.format(
        history=conversation_history,
        context=context,
        text=text
    )

    rag_chain = (
        {"context":RunnablePassthrough(),"text":RunnablePassthrough()}
         | custom_prompt
         | model
         | StrOutputParser()
    )

    response = rag_chain.invoke(formatted_custom_prompt)

    print(response)

    return {'chat_room_id' : chat_room_id,'message':response }
'''
    # 현재 대화와 AI 응답을 메모리에 추가
    memory.save_context({"user": text}, {"ai": response})

    output_buffer = text_to_speak(response)

    # 저장된 파일을 읽어서 클라이언트에게 청크 단위로 스트리밍
    def iterfile(output_buffer):
        chunk_size = 1024
        try:
            while True:
                chunk = output_buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            # 예외 발생 시 로그 출력 또는 처리
            print(f"Error while streaming file: {e}")
            # 빈 generator 반환하여 종료
            yield b""

    return StreamingResponse(iterfile(output_buffer), media_type="audio/wav")'''
    
