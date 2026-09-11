import os
import tempfile

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

# 우리가 만든 LangServe + ngrok 주소
LANGSERVE_URL = (
    "https://zebra-attendee-koala.ngrok-free.dev/llm/"
)

# 개인화 Llama3를 원격 호출
personal_llm = RemoteRunnable(
    LANGSERVE_URL
)


# =========================================================
# 1. Gemini / LangChain 응답에서 텍스트만 추출
# =========================================================

def extract_response_text(content):

    # 일반 문자열
    if isinstance(content, str):
        return content

    # Gemini 최신 content block
    if isinstance(content, list):

        texts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        texts.append(text)

            elif hasattr(item, "text"):

                text = item.text

                if text:
                    texts.append(text)

        return "\n".join(
            texts
        ).strip()

    # AIMessage 등
    if hasattr(content, "content"):

        return extract_response_text(
            content.content
        )

    return str(content)


# =========================================================
# 2. 토큰 길이 계산
# =========================================================

def tiktoken_len(text):

    tokenizer = tiktoken.get_encoding(
        "cl100k_base"
    )

    tokens = tokenizer.encode(text)

    return len(tokens)


# =========================================================
# 3. 업로드 문서 읽기
# PDF / DOCX / PPTX
# =========================================================

def get_text(docs):

    doc_list = []

    for doc in docs:

        file_name = doc.name

        suffix = os.path.splitext(
            file_name
        )[1]

        # 업로드 파일 임시 저장
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                doc.getvalue()
            )

            temp_path = (
                temp_file.name
            )

        logger.info(
            f"문서 처리 시작: {file_name}"
        )

        try:

            # PDF
            if file_name.lower().endswith(
                ".pdf"
            ):

                loader = PyPDFLoader(
                    temp_path
                )

                documents = (
                    loader.load()
                )

            # Word
            elif file_name.lower().endswith(
                ".docx"
            ):

                loader = Docx2txtLoader(
                    temp_path
                )

                documents = (
                    loader.load()
                )

            # PowerPoint
            elif file_name.lower().endswith(
                ".pptx"
            ):

                loader = (
                    UnstructuredPowerPointLoader(
                        temp_path
                    )
                )

                documents = (
                    loader.load()
                )

            else:
                continue

            # 실제 파일명을 출처로 저장
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
# 4. 문서 Chunk 분할
# =========================================================

def get_text_chunks(text):

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len,
        )
    )

    chunks = (
        text_splitter
        .split_documents(text)
    )

    return chunks


# =========================================================
# 5. FAISS VectorStore 생성
# =========================================================

def get_vectorstore(text_chunks):

    embeddings = (
        HuggingFaceEmbeddings(

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
    )

    vectordb = (
        FAISS.from_documents(
            text_chunks,
            embeddings
        )
    )

    return vectordb


# =========================================================
# 6. 검색 문서를 Context 문자열로 변환
# =========================================================

def format_documents(docs):

    context = ""

    for i, doc in enumerate(docs):

        source = (
            doc.metadata.get(
                "source",
                "알 수 없음"
            )
        )

        page = (
            doc.metadata.get(
                "page",
                None
            )
        )

        page_text = ""

        if page is not None:

            page_text = (
                f"페이지: "
                f"{page + 1}\n"
            )

        context += f"""

[참고 문서 {i + 1}]

출처: {source}

{page_text}

{doc.page_content}

"""

    return context


# =========================================================
# 7. 개인화 질문 판별
# =========================================================

def is_personal_question(question):

    personal_keywords = [

        "내 이름",
        "이름이",
        "이름은",

        "학번",

        "전화번호",
        "휴대폰",
        "연락처",

        "아버지",
        "어머니",
        "부모님",

        "주소",
        "거주지",
        "거주 지역",

        "소속 대학교",
        "소속대학교",
        "학교 어디",
        "대학교 어디",

        "내 학력",
        "최종 학력",

        "내 전공",

        "생년월일",

        "출생지",

        "고향",

        "취미",

        "특기",

        "자격증",
    ]

    question_lower = (
        question.lower()
    )

    return any(
        keyword.lower()
        in question_lower

        for keyword
        in personal_keywords
    )


# =========================================================
# 8. llama3-personal 일반 호출
# =========================================================

def ask_personal_llm(question):

    response = personal_llm.invoke(
        question
    )

    return extract_response_text(
        response
    )


# =========================================================
# 9. Gemini 일반 질문
# =========================================================

def ask_general_gemini(
    question,
    google_api_key
):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.3,
        google_api_key=google_api_key,
    )

    history_text = ""

    for message in (
        st.session_state
        .messages[-6:]
    ):

        if (
            message["role"]
            == "user"
        ):

            history_text += (
                "\n사용자: "
                + message["content"]
            )

        elif (
            message["role"]
            == "assistant"
        ):

            history_text += (
                "\nAI: "
                + message["content"]
            )

    prompt = f"""
당신은 친절하고 정확한 AI 어시스턴트입니다.

사용자의 질문에 자연스러운 한국어로 답하세요.

[이전 대화]
{history_text}

[사용자 질문]
{question}

[답변]
"""

    response = llm.invoke(
        prompt
    )

    return extract_response_text(
        response.content
    )


# =========================================================
# 10. 로컬 Llama3 일반 질문
# =========================================================

def ask_general_local(question):

    prompt = f"""
다음 질문에 정확하고 자연스러운 한국어로 답하세요.

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
# 11. Gemini + RAG
# =========================================================

def ask_gemini_rag(
    question,
    vectordb,
    google_api_key
):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    retriever = (
        vectordb.as_retriever(

            search_type=(
                "similarity"
            ),

            search_kwargs={
                "k": 5
            },
        )
    )

    source_documents = (
        retriever.invoke(
            question
        )
    )

    context = format_documents(
        source_documents
    )

    prompt = f"""
당신은 업로드된 문서를 분석하여
질문에 답하는 RAG 챗봇입니다.

아래 참고 문서를 자세히 확인한 뒤
사용자의 질문에 답하세요.

규칙:

1. 참고 문서에서 질문과 관련된 내용을 먼저 찾으세요.
2. 관련 정보가 있으면 문서 내용을 근거로 정확하게 답하세요.
3. 여러 문서에 정보가 나뉘어 있으면 종합해서 답하세요.
4. 관련 정보가 정말 없을 때만
   "업로드된 문서에서 해당 정보를 찾을 수 없습니다."
   라고 답하세요.
5. 문서에 없는 내용을 임의로 만들어내지 마세요.
6. 한국어로 자연스럽게 답하세요.

[참고 문서]

{context}

[사용자 질문]

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
# 12. llama3-personal + RAG
# Gemini API Key가 없을 때 사용
# =========================================================

def ask_local_rag(
    question,
    vectordb
):

    retriever = (
        vectordb.as_retriever(

            search_type=(
                "similarity"
            ),

            search_kwargs={
                "k": 5
            },
        )
    )

    source_documents = (
        retriever.invoke(
            question
        )
    )

    context = format_documents(
        source_documents
    )

    prompt = f"""
당신은 문서 기반 질의응답 AI입니다.

아래 참고 문서를 읽고
질문에 답하세요.

문서에 답이 있는 경우에는
반드시 문서 내용을 우선 사용하세요.

문서에 없는 내용을
임의로 만들어내지 마세요.

[참고 문서]

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

    return (
        answer,
        source_documents
    )


# =========================================================
# 13. 참고 문서 출력
# =========================================================

def show_source_documents(
    source_documents
):

    with st.expander(
        "📑 참고 문서 확인"
    ):

        for i, doc in enumerate(
            source_documents
        ):

            source = (
                doc.metadata.get(
                    "source",
                    "알 수 없음"
                )
            )

            page = (
                doc.metadata.get(
                    "page",
                    None
                )
            )

            st.markdown(
                f"### 참고 문서 "
                f"{i + 1}"
            )

            st.write(
                f"출처: {source}"
            )

            if page is not None:

                st.write(
                    f"페이지: "
                    f"{page + 1}"
                )

            st.write(
                doc.page_content
            )

            st.divider()


# =========================================================
# 14. Streamlit Main
# =========================================================

def main():

    st.set_page_config(
        page_title=(
            "Personal AI RAG"
        ),
        page_icon="🤖",
        layout="wide",
    )

    st.title(
        "🤖 Personal AI + RAG Chat"
    )

    st.info(
        "Google API Key는 선택사항입니다. "
        "API Key가 있으면 Gemini를 사용하고, "
        "없으면 개인화 Llama3를 사용합니다. "
        "PDF/DOCX/PPTX도 업로드할 수 있습니다."
    )


    # -----------------------------------------------------
    # Session State
    # -----------------------------------------------------

    if (
        "messages"
        not in st.session_state
    ):

        st.session_state.messages = []


    if (
        "vectordb"
        not in st.session_state
    ):

        st.session_state.vectordb = None


    if (
        "processComplete"
        not in st.session_state
    ):

        st.session_state.processComplete = (
            False
        )


    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    with st.sidebar:

        st.header(
            "📂 문서 업로드"
        )

        uploaded_files = (
            st.file_uploader(

                "PDF / DOCX / PPTX",

                type=[
                    "pdf",
                    "docx",
                    "pptx",
                ],

                accept_multiple_files=True,
            )
        )

        st.divider()

        st.subheader(
            "🔑 Gemini API"
        )

        google_api_key = (
            st.text_input(
                "Google API Key "
                "(선택사항)",

                type="password",

                key=(
                    "chatbot_api_key"
                ),
            )
        )

        if google_api_key:

            st.success(
                "Gemini 모드 사용 가능"
            )

        else:

            st.info(
                "API Key 없음 → "
                "llama3-personal 사용"
            )

        st.divider()

        process = st.button(
            "Process",
            use_container_width=True,
        )


    # -----------------------------------------------------
    # 문서 Process
    # -----------------------------------------------------

    if process:

        if not uploaded_files:

            st.warning(
                "먼저 문서를 "
                "업로드해 주세요."
            )

        else:

            with st.spinner(
                "문서를 분석하고 "
                "벡터DB를 생성 중입니다..."
            ):

                docs = get_text(
                    uploaded_files
                )

                if len(docs) == 0:

                    st.error(
                        "읽을 수 있는 "
                        "텍스트가 없습니다."
                    )

                    st.stop()

                text_chunks = (
                    get_text_chunks(
                        docs
                    )
                )

                vectordb = (
                    get_vectorstore(
                        text_chunks
                    )
                )

                st.session_state.vectordb = (
                    vectordb
                )

                st.session_state.processComplete = (
                    True
                )

            st.success(
                f"문서 처리 완료! "
                f"{len(docs)}개 페이지/요소, "
                f"{len(text_chunks)}개 Chunk"
            )


    # -----------------------------------------------------
    # 최초 AI 메시지
    # -----------------------------------------------------

    if (
        len(
            st.session_state.messages
        )
        == 0
    ):

        st.session_state.messages.append(
            {
                "role": "assistant",

                "content":
                    "안녕하세요! 👋\n\n"
                    "• 개인화 질문\n"
                    "• 일반 질문\n"
                    "• 업로드 문서 질문\n\n"
                    "모두 가능합니다."
            }
        )


    # -----------------------------------------------------
    # 기존 채팅 출력
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # 질문 입력
    # -----------------------------------------------------

    query = st.chat_input(
        "질문을 입력해 주세요."
    )


    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message(
            "user"
        ):

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

                    source_documents = None


                    # =====================================
                    # A. 개인화 질문
                    # =====================================

                    if is_personal_question(
                        query
                    ):

                        answer = (
                            ask_personal_llm(
                                query
                            )
                        )

                        st.caption(
                            "🦙 "
                            "Personal Llama3"
                        )


                    # =====================================
                    # B. 문서가 Process 되어 있음
                    # =====================================

                    elif (
                        st.session_state
                        .processComplete
                    ):

                        # Gemini API Key 있음
                        if google_api_key:

                            (
                                answer,
                                source_documents
                            ) = ask_gemini_rag(

                                query,

                                st.session_state
                                .vectordb,

                                google_api_key,
                            )

                            st.caption(
                                "📚 "
                                "RAG + Gemini"
                            )


                        # Gemini API Key 없음
                        else:

                            (
                                answer,
                                source_documents
                            ) = ask_local_rag(

                                query,

                                st.session_state
                                .vectordb,
                            )

                            st.caption(
                                "📚 "
                                "RAG + "
                                "Personal Llama3"
                            )


                    # =====================================
                    # C. 문서 없음 → 일반 질문
                    # =====================================

                    else:

                        # Gemini 있음
                        if google_api_key:

                            answer = (
                                ask_general_gemini(
                                    query,
                                    google_api_key
                                )
                            )

                            st.caption(
                                "✨ Gemini"
                            )


                        # Gemini 없음
                        else:

                            answer = (
                                ask_general_local(
                                    query
                                )
                            )

                            st.caption(
                                "🦙 "
                                "Personal Llama3"
                            )


                    # -------------------------------------
                    # 답변 출력
                    # -------------------------------------

                    st.markdown(
                        answer
                    )


                    # -------------------------------------
                    # 참고 문서 표시
                    # -------------------------------------

                    if source_documents:

                        show_source_documents(
                            source_documents
                        )


                    # -------------------------------------
                    # 답변 저장
                    # -------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )


                except Exception as e:

                    st.error(
                        "답변 생성 중 "
                        f"오류가 발생했습니다:\n\n{e}"
                    )

                    st.info(
                        "Google API Key를 사용하지 않는 경우 "
                        "PC에서 Ollama, LangServe, ngrok이 "
                        "모두 실행 중인지 확인하세요."
                    )


# =========================================================
# 프로그램 시작
# =========================================================

if __name__ == "__main__":
    main()
