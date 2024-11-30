# 책 파일을 받아서 임베딩 데이터베이스를 생성하는 코드
from fastapi import APIRouter
from fastapi import File,UploadFile,HTTPException
import os
import shutil
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import io
import tempfile

EMB_DB_PATH = './emb_db'


emb_router = APIRouter(prefix='/emb')

emb = OpenAIEmbeddings()


# 삽입된 txt 파일 벡터화 및 저장
@emb_router.post("/vectorized")
async def text_vectorize(book_file : UploadFile = File(...)):
    # pdf 파일인지 txt 파일인지 확인하기
    if book_file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    try:
        content = await book_file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail="File read error")

    docs=[]

    try:
        if book_file.content_type == 'text/plain':
            content = content.decode('utf-8')
            doc = Document(page_content=content)
            docs.append(doc)
        elif book_file.content_type == 'application/pdf':
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            pdf_loader = PyPDFLoader(tmp_file_path)
            pdf_docs = pdf_loader.load()
            docs.extend(pdf_docs)
            os.remove(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")

    # 책 제목 추출해서 db 이름으로 사용하기
    # 파일 이름에서 확장자 제거하여 디렉터리 이름 생성
    file_name_without_ext = os.path.splitext(book_file.filename)[0]
    persist_directory = os.path.join("./emb_db", file_name_without_ext)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # 각 조각의 최대 문자 수를 1000자로 설정합니다.
        chunk_overlap = 100  # 각 조각이 서로 겹치는 부분을 100자로 설정하여 분할된 텍스트 간의 연결성을 유지합니다.
    )
    docs_spliter = text_splitter.split_documents(docs)

    try:
        openai_vectorstore = Chroma.from_documents(
            docs_spliter,
            emb,
            persist_directory=persist_directory
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Embedding storage error")

    return {"message": f"File processed and vectorized successfully in {persist_directory}"}

# 파일 입력 받으면 내용 저장하지 말고 그냥 바로 임베딩 해보기