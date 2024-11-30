from fastapi import APIRouter
from fastapi import File,UploadFile
import os
import shutil
from langchain.docstore.document import Document
#from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
import csv

load_dotenv()
text_router = APIRouter(prefix="/text")

TXT_PATH = "txt_files"
CSV_PATH = "csv_files"
OPENAI_DB_PATH = "./openai_db"
LLAMA_DB_PATH = "./llama_db"
docs=[]

#emb = HuggingFaceEndpointEmbeddings()
'''
llama_emb =OllamaEmbeddings(
    model="mxbai-embed-large",# 임베딩 모델 지정(ollama 홈페이지에서 찾음)
    num_gpu=1,
    show_progress=True
)
'''
emb = OpenAIEmbeddings()
if not os.path.exists(TXT_PATH):
    os.mkdir(TXT_PATH)
# txt 파일을 삽입하면 자동으로 벡터화 임베딩 되도록 만들기

# txt 파일 삽입 
@text_router.post("/upload_txt")
async def upload_txt(file : UploadFile = File(...)):
    save_path = TXT_PATH + '/' + file.filename

    with open(save_path,"wb") as buffer: # wb = 바이너리 형식 쓰기모드
        shutil.copyfileobj(file.file, buffer)
    return {"response":f"{save_path}에 저장 되었습니다."}

# 삽입된 txt 파일 벡터화 및 저장
@text_router.post("/vectorized_txt")
def text_vectorize():
    all_files = os.listdir(TXT_PATH)
    #matching_file = next((file for file in all_files if file ==f"{key}.txt"), None)
    for file_path in all_files:
        with open(f"txt_files/{file_path}", 'r', encoding='utf-8') as file:
            content = file.read()
            doc = Document(page_content=content)
            docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # 각 조각의 최대 문자 수를 1000자로 설정합니다.
    chunk_overlap = 100  # 각 조각이 서로 겹치는 부분을 100자로 설정하여 분할된 텍스트 간의 연결성을 유지합니다.
    )
    docs_spliter = text_splitter.split_documents(docs)

    openai_vectorstore = Chroma.from_documents(
    docs_spliter,
    emb,
    #persist_directory=OPENAI_DB_PATH
    persist_directory=LLAMA_DB_PATH
    )

    return all_files,docs_spliter[:10]
    

@text_router.get("/vectorcized_embedding")
def total_embadding():
    all_txt_files = os.listdir(TXT_PATH)
    all_csv_files = os.listdir(CSV_PATH)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # 각 조각의 최대 문자 수를 1000자로 설정합니다.
    chunk_overlap = 50  # 각 조각이 서로 겹치는 부분을 100자로 설정하여 분할된 텍스트 간의 연결성을 유지합니다.
    )

    for file_path in all_txt_files:
        with open(f"txt_files/{file_path}", 'r', encoding='utf-8') as file:
            content = file.read()
            doc = Document(page_content=content)
            docs.append(doc)
    print(len(docs))

    
    for file_path in all_csv_files:
         with open(f"csv_files/{file_path}", 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                content = ' '.join(row) 
                doc = Document(page_content=content)
                docs.append(doc)

    print(len(docs))

    docs_spliter = text_splitter.split_documents(docs)

    openai_vectorstore = Chroma.from_documents(
    docs_spliter,
    emb,
    #persist_directory=OPENAI_DB_PATH
    persist_directory=OPENAI_DB_PATH
    )

    return docs_spliter[:10]


@text_router.post("/best_kid_vectorized_txt")
def text_vectorize():
    all_files = os.listdir('kid_best_txt')

    for file_path in all_files:
        with open(f"kid_best_txt/{file_path}", 'r', encoding='utf-8') as file:
            content = file.read()
            doc = Document(page_content=content)
            docs.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # 각 조각의 최대 문자 수를 1000자로 설정합니다.
    chunk_overlap = 50  # 각 조각이 서로 겹치는 부분을 100자로 설정하여 분할된 텍스트 간의 연결성을 유지합니다.
    )
    docs_spliter = text_splitter.split_documents(docs)

    kid_vector_store= Chroma.from_documents(
    docs_spliter,
    emb,
    #persist_directory=OPENAI_DB_PATH
    persist_directory='./kid_db'
    )

    return docs_spliter[:10]
    