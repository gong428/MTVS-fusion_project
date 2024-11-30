from fastapi import APIRouter
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

from dotenv import load_dotenv
load_dotenv()
#임베딩 저장 장소
OPENAI_DB_PATH = "./openai_db"
#emb = HuggingFaceEndpointEmbeddings()
#llama_emb = OllamaEmbeddings(model="mxbai-embed-large",# 임베딩 모델 지정(ollama 홈페이지에서 찾음)num_gpu=1,show_progress=True)


# 임베딩 도구 세팅
emb = OpenAIEmbeddings()
# 모델 설정
model = ChatOpenAI(model_name="gpt-4o-2024-08-06")

# 입력 형식 선언
class Prompt(BaseModel):
    userNo : int
    trigger: str
    chattingContents : str


# 트리거 저장소
prompt_dic = {}
# 대화 내용 저장소
memory_store = {}
#model = ChatOllama(model="llama3.1:latest")

system_prompt = """
    너는 8~12세의 어린이에게 교내에서 생길 수 있는 안전 사고들과 안전 사고별 통계를 설명해주는 어시스턴트이고 이름은 안전이야.
    아래 지침과 Context에 내용을 참조해서 사용자 요청별로 다르게 답변해줘.
    1. 사용자가 Trigger행위를 하면 안되는 이유를 요청하면 Trigger로 인해 발생할 수 있는 사고와 사고로 생길 수 있는 부상명 Context를 참조하여 답변해줘.
    2. 사용자가 부상에 대한 설명을 요구하면 부상에 증상과 치료 방법과 최소 수술 비용과 최대 수술 비용을 답변해줘.
    3. 수술비용으로 Kid_Context를 참조해서 최대로 구매 가능한  초등학생 인기 게임,초등학생 인기 장난감을 랜덤하게 조합해서 답변해줘.
    4. 모든 답변은 8~12세의 어린이가 이해하기 쉬운 문법으로 답변해줘.
    5. 모든 답변은 ~해, ~하자, ~까 등 반말 체로 답변해줘
    6. 위의 모든 답변은 500자 이내로 생성해줘

    검색된 Context 기반으로 Question 에 한국어로 대답을 하도록 해. 
    Question : {text} 
    Context : {context} 
    Trigger : {trigger}
    Kid_Context : {kid_context}

    Answer:
"""

chatbot_router = APIRouter(prefix="/chatbot")

#각각의 문서 내용을 줄바꿈으로 구분된 하나의 문자열로 반환하는 역할을 합니다.
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



#벡터화 db 불러와서 답변 생성하는 챗봇

@chatbot_router.post("/from_vec",tags=['Chatbot'])
def chatbot_model4(prompt_model: Prompt):
    user_num = prompt_model.userNo
    # 대화 기록 저장소 생성
    if user_num not in memory_store:
        memory_store[user_num] = ConversationBufferMemory()

    memory = memory_store[user_num]

    # 임베딩 db 불러오기
    openai_db = Chroma(
        persist_directory=OPENAI_DB_PATH,
        embedding_function=emb
    )

    kid_db = Chroma(
        persist_directory='./kid_db',
        embedding_function=emb
    )
    #임베딩 db 를 참조한 chattingContents(유저 입력) 유사한 내용 5개까지 찾기
    openai_retriever = openai_db.as_retriever(
        search_kwargs={"k":5}
    )
    kid_retriever = kid_db.as_retriever(
        search_kwargs={"k":5}
    )
    #프롬프트에 삽입할 history 형태 만들어주기
    conversation_history = memory.load_memory_variables({})

    # 시스템 프롬프트에서 참조할 context 
    context = openai_retriever.invoke(prompt_model.chattingContents)
    kid_context = kid_retriever.invoke(prompt_model.chattingContents)
    #
    custom_prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # 필요한 모든 변수를 템플릿에 전달(질문용 템플릿 작성)
    formatted_custom_prompt = custom_prompt.format(
        history=conversation_history,
        context=context,
        text=prompt_model.chattingContents,
        trigger = prompt_model.trigger,
        kid_context = kid_context

    )

    rag_chain = (
        {"context":openai_retriever | format_docs,"text":RunnablePassthrough(),"trigger":RunnablePassthrough(),"kid_context":RunnablePassthrough()}
         | custom_prompt
         | model
         | StrOutputParser()
    )

    response = rag_chain.invoke(formatted_custom_prompt)

    # 현재 대화와 AI 응답을 메모리에 추가
    memory.save_context({"user": prompt_model.chattingContents}, {"ai": response})

    # 출력값
    output_json = {
        "userNo" : user_num,
        "trigger" :  prompt_model.trigger,
        "chattingContents": response
        }

    return output_json

## 토론내용 요약하는 ai 만들기

## 책내용 기반 자유 토론 주제 생성 ai 만들기