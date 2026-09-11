import os
import tempfile
import re

import streamlit as st
import tiktoken

from loguru import logger

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langserve import RemoteRunnable


# =========================================================
# 설정
# =========================================================

LANGSERVE_URL = (
    "https://zebra-attendee-koala.ngrok-free.dev/llm/"
)

personal_llm = RemoteRunnable(
    LANGSERVE_URL
)


# =========================================================
# 응답에서 텍스트만 추출
# =========================================================

def extract_response_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        texts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get("text", "")

                    if text:
                        texts.append(text)

            elif hasattr(item, "text"):

                text = item.text

                if text:
                    texts.append(text)

        return "\n".join(texts).strip()

    if hasattr(content, "content"):

        return extract_response_text(
            content.content
        )

    return str(content)


# =========================================================
# 일반 질문의 이상한 개인화 말투 정리
# =========================================================

def clean_general_answer(answer):

    answer = answer.strip()

    # 미세조정 모델이 일반 질문에서도
    # "제 대한민국..." 같은 패턴을 출력하는 경우 보정
    replacements = {
        "제대한민국": "대한민국",
        "제 대한민국": "대한민국",
        "저의 대한민국": "대한민국",
        "내 대한민국": "대한민국",
    }

    for old, new in replacements.items():

        answer = answer.replace(
            old,
            new
        )

    return answer


# =========================================================
# 토큰 길이
# =========================================================

def tiktoken_len(text):

    tokenizer = tiktoken.get_encoding(
        "cl100k_base"
    )

    return len(
        tokenizer.encode(text)
    )


# =========================================================
# 문서 읽기
# =========================================================

def get_text(docs):

    doc_list = []

    for doc in docs:

        file_name = doc.name

        suffix = os.path.splitext(
            file_name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                doc.getvalue()
            )

            temp_path = temp_file.name

        logger.info(
            f"문서 처리 시작: {file_name}"
        )

        try:

            if file_name.lower().endswith(
                ".pdf"
            ):

                loader = PyPDFLoader(
                    temp_path
                )

                documents = loader.load()

            elif file_name.lower().endswith(
                ".docx"
            ):

                loader = Docx2txtLoader(
                    temp_path
                )

                documents = loader.load()

            elif file_name.lower().endswith(
                ".pptx"
            ):

                loader = (
                    UnstructuredPowerPointLoader(
                        temp_path
                    )
                )

                documents = loader.load()

            else:

                continue


            for document in documents:

                document.metadata[
                    "source"
                ] = file_name


            doc_list.extend(
                documents
            )

        finally:

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass

    return doc_list


# =========================================================
# 문서 분할
# =========================================================

def get_text_chunks(text):

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len,
        )
    )

    return (
        text_splitter
        .split_documents(text)
    )


# =========================================================
# FAISS
# =========================================================

def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(

        model_name=(
            "jhgan/"
            "ko-sroberta-multitask"
        ),

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings":
                True
        },
    )

    return FAISS.from_documents(
        text_chunks,
        embeddings
    )


# =========================================================
# 검색 문서 → Context
# =========================================================

def format_documents(docs):

    context = ""

    for i, doc in enumerate(docs):

        source = doc.metadata.get(
            "source",
            "알 수 없음"
        )

        page = doc.metadata.get(
            "page",
            None
        )

        page_text = ""

        if page is not None:

            page_text = (
                f"페이지: {page + 1}"
            )

        context += f"""

[문서 {i + 1}]
출처: {source}
{page_text}

{doc.page_content}

"""

    return context


# =========================================================
# 개인화 질문 판별
# =========================================================

def is_personal_question(question):

    keywords = [

        "내 이름",
        "제 이름",
        "이름이 뭐",
        "이름은 뭐",

        "내 학번",
        "제 학번",
        "학번은",

        "내 전화번호",
        "제 전화번호",
        "전화번호는",
        "휴대폰 번호",
        "연락처",

        "내 주소",
        "제 주소",
        "거주 지역",
        "거주지",

        "내 출생지",
        "제 출생지",
        "출생지는",

        "내 생년월일",
        "제 생년월일",

        "내 학교",
        "제 학교",
        "소속대학교",
        "소속 대학교",

        "내 전공",
        "제 전공",

        "내 취미",
        "제 취미",

        "내 자격증",
        "제 자격증",

        "아버지 이름",
        "어머니 이름",
        "부모님 이름",
    ]

    q = question.lower()

    return any(
        keyword.lower() in q
        for keyword in keywords
    )


# =========================================================
# 개인화 질문
# =========================================================

def ask_personal_llm(question):

    prompt = f"""
다음 질문은 사용자의 개인 정보에 관한 질문입니다.

당신이 학습한 개인화 데이터를 기준으로
질문에 직접적으로 답하세요.

불필요한 일반 지식 설명은 하지 마세요.

질문:
{question}

답변:
"""

    response = personal_llm.invoke(
        prompt
    )

    return extract_response_text(
        response
    )


# =========================================================
# 일반 질문 - Local Llama3
# =========================================================

def ask_general_local(question):

    prompt = f"""
당신은 일반 지식 질문에 답하는 AI입니다.

중요한 규칙:

1. 이 질문은 사용자의 개인정보를 묻는 질문이 아닙니다.
2. 일반적인 객관적 사실을 답하세요.
3. 절대로 '제', '저의', '내', '나는' 등의
   1인칭 표현으로 사실을 설명하지 마세요.
4. 개인화 학습 데이터의 문장 형식을 따라 하지 마세요.
5. 질문에 간결하고 자연스러운 한국어로 답하세요.

예시:

질문: 한국의 수도는?
좋은 답변: 대한민국의 수도는 서울입니다.

나쁜 답변:
제 대한민국의 수도는 서울입니다.
제 수도는 서울입니다.

질문:
{question}

답변:
"""

    response = personal_llm.invoke(
        prompt
    )

    answer = extract_response_text(
        response
    )

    return clean_general_answer(
        answer
    )


# =========================================================
# 일반 질문 - Gemini
# =========================================================

def ask_general_gemini(
    question,
    google_api_key
):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2,
        google_api_key=google_api_key,
    )

    prompt = f"""
다음 질문에 정확하고 자연스러운
한국어로 답하세요.

질문:
{question}

답변:
"""

    response = llm.invoke(
        prompt
    )

    return extract_response_text(
        response.content
    )


# =========================================================
# RAG 검색
# =========================================================

def retrieve_documents(
    question,
    vectordb
):

    retriever = vectordb.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 5
        },
    )

    return retriever.invoke(
        question
    )


# =========================================================
# RAG + Gemini
# =========================================================

def ask_gemini_rag(
    question,
    vectordb,
    google_api_key
):

    source_documents = (
        retrieve_documents(
            question,
            vectordb
        )
    )

    context = format_documents(
        source_documents
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    prompt = f"""
아래에는 업로드된 문서에서
검색한 내용이 있습니다.

먼저 검색된 문서가 사용자의 질문과
실제로 관련이 있는지 판단하세요.

관련이 있다면:
- 문서의 내용을 우선하여 답하세요.
- 문서에 있는 사실을 정확하게 사용하세요.

관련이 없다면:
- 문서 내용을 억지로 사용하지 말고
  일반 지식을 이용하여 질문에 답하세요.

문서에 없는 사실을
문서에서 찾았다고 말해서는 안 됩니다.

[검색된 문서]
{context}

[질문]
{question}

[답변]
"""

    response = llm.invoke(
        prompt
    )

    answer = extract_response_text(
        response.content
    )

    return (
        answer,
        source_documents
    )


# =========================================================
# RAG + Personal Llama3
# =========================================================

def ask_local_rag(
    question,
    vectordb
):

    source_documents = (
        retrieve_documents(
            question,
            vectordb
        )
    )

    context = format_documents(
        source_documents
    )

    prompt = f"""
당신은 문서 검색 기능이 있는 AI입니다.

아래는 업로드된 문서에서
질문과 관련성이 높은 부분을 검색한 결과입니다.

먼저 검색 내용이 질문과
실제로 관련 있는지 판단하세요.

관련 있다면:
문서의 내용을 근거로 답하세요.

관련이 없다면:
검색 내용을 억지로 사용하지 말고
일반 지식을 이용하여 답하세요.

일반 지식으로 답할 때에는
'제', '저의', '내' 등의
개인화된 1인칭 표현을 사용하지 마세요.

[검색 문서]
{context}

[질문]
{question}

[답변]
"""

    response = personal_llm.invoke(
        prompt
    )

    answer = extract_response_text(
        response
    )

    answer = clean_general_answer(
        answer
    )

    return (
        answer,
        source_documents
    )


# =========================================================
# Streamlit
# =========================================================

def main():

    st.set_page_config(
        page_title="RAG Chat",
        page_icon="📚",
        layout="wide",
    )


    # -------------------------
    # Session
    # -------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []


    if "vectordb" not in st.session_state:

        st.session_state.vectordb = None


    if "processComplete" not in st.session_state:

        st.session_state.processComplete = False


    # =====================================================
    # Sidebar
    # 교수님 예제처럼 최소한만 표시
    # =====================================================

    with st.sidebar:

        uploaded_files = st.file_uploader(

            "Upload your file",

            type=[
                "pdf",
                "docx",
                "pptx"
            ],

            accept_multiple_files=True,
        )


        process = st.button(
            "Process"
        )


        google_api_key = st.text_input(

            "Google API Key (optional)",

            type="password"
        )


    # =====================================================
    # Process
    # =====================================================

    if process:

        if uploaded_files:

            with st.spinner(
                "Processing..."
            ):

                docs = get_text(
                    uploaded_files
                )

                chunks = get_text_chunks(
                    docs
                )

                vectordb = get_vectorstore(
                    chunks
                )

                st.session_state.vectordb = (
                    vectordb
                )

                st.session_state.processComplete = (
                    True
                )


    # =====================================================
    # 최초 메시지
    # =====================================================

    if not st.session_state.messages:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    "안녕하세요. 무엇이 궁금하신가요?"
            }
        )


    # =====================================================
    # 기존 대화
    # =====================================================

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # =====================================================
    # 입력
    # =====================================================

    query = st.chat_input(
        "메시지를 입력해 주세요"
    )


    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )


        with st.chat_message("user"):

            st.markdown(
                query
            )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    # -------------------------------------
                    # 1. 개인화 질문
                    # -------------------------------------

                    if is_personal_question(
                        query
                    ):

                        answer = (
                            ask_personal_llm(
                                query
                            )
                        )


                    # -------------------------------------
                    # 2. 문서가 있음
                    # -------------------------------------

                    elif (
                        st.session_state
                        .processComplete
                    ):

                        # Gemini Key 있음
                        if google_api_key:

                            answer, _ = (
                                ask_gemini_rag(
                                    query,

                                    st.session_state
                                    .vectordb,

                                    google_api_key,
                                )
                            )


                        # Gemini Key 없음
                        else:

                            answer, _ = (
                                ask_local_rag(
                                    query,

                                    st.session_state
                                    .vectordb,
                                )
                            )


                    # -------------------------------------
                    # 3. 문서 없음
                    # -------------------------------------

                    else:

                        # Gemini 사용
                        if google_api_key:

                            answer = (
                                ask_general_gemini(
                                    query,
                                    google_api_key
                                )
                            )


                        # 개인 Llama3 일반 질문
                        else:

                            answer = (
                                ask_general_local(
                                    query
                                )
                            )


                    # -------------------------------------
                    # 출력
                    # -------------------------------------

                    st.markdown(
                        answer
                    )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )


                except Exception as e:

                    st.error(
                        f"오류가 발생했습니다: {e}"
                    )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":
    main()
