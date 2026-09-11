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


# ---------------------------------------------------------
# 1. 토큰 길이 계산
# 교수님 PDF 93~96라인과 같은 역할
# ---------------------------------------------------------
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# ---------------------------------------------------------
# 2. 업로드 문서 읽기
# PDF / DOCX / PPTX
# 교수님 PDF 98~118라인 역할
# ---------------------------------------------------------
def get_text(docs):

    doc_list = []

    for doc in docs:

        file_name = doc.name

        # 업로드 파일을 임시 파일로 저장
        suffix = os.path.splitext(file_name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(doc.getvalue())
            temp_path = temp_file.name

        logger.info(f"문서 처리 시작: {file_name}")

        try:

            if file_name.lower().endswith(".pdf"):

                loader = PyPDFLoader(temp_path)
                documents = loader.load()

            elif file_name.lower().endswith(".docx"):

                loader = Docx2txtLoader(temp_path)
                documents = loader.load()

            elif file_name.lower().endswith(".pptx"):

                loader = UnstructuredPowerPointLoader(temp_path)
                documents = loader.load()

            else:

                continue

            # 출처를 실제 파일명으로 표시
            for document in documents:
                document.metadata["source"] = file_name

            doc_list.extend(documents)

        finally:

            try:
                os.remove(temp_path)
            except:
                pass

    return doc_list


# ---------------------------------------------------------
# 3. 문서 Chunk 분할
# 교수님 PDF 121~128라인 역할
# chunk_size = 900
# chunk_overlap = 100
# ---------------------------------------------------------
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len,
    )

    chunks = text_splitter.split_documents(text)

    return chunks


# ---------------------------------------------------------
# 4. FAISS VectorStore 생성
# 교수님 PDF 131~139라인 역할
# ---------------------------------------------------------
def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    vectordb = FAISS.from_documents(
        text_chunks,
        embeddings
    )

    return vectordb


# ---------------------------------------------------------
# 5. 검색된 문서 내용을 하나의 Context로 변환
# ---------------------------------------------------------
def format_documents(docs):

    context = ""

    for i, doc in enumerate(docs):

        context += f"""

[참고 문서 {i + 1}]
출처: {doc.metadata.get("source", "알 수 없음")}

{doc.page_content}

"""

    return context


# ---------------------------------------------------------
# 6. Gemini에게 RAG 질문
# ---------------------------------------------------------
def ask_gemini(question, vectordb, google_api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    # 교수님 PDF와 동일하게 MMR 검색
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
        },
    )

    source_documents = retriever.invoke(question)

    context = format_documents(source_documents)

    # 이전 대화 내용도 일부 포함
    history_text = ""

    for message in st.session_state.messages[-6:]:

        if message["role"] == "user":
            history_text += f"\n사용자: {message['content']}"

        elif message["role"] == "assistant":
            history_text += f"\nAI: {message['content']}"

    prompt = f"""
다음 참고 문서의 내용을 바탕으로 사용자의 질문에 답하세요.

반드시 다음 원칙을 지키세요.

1. 참고 문서에 근거하여 답변하세요.
2. 문서에서 찾을 수 없는 내용은 임의로 만들어내지 마세요.
3. 문서에 해당 정보가 없다면
   "업로드된 문서에서 해당 정보를 찾을 수 없습니다."
   라고 답하세요.
4. 답변은 자연스러운 한국어로 작성하세요.

[이전 대화]
{history_text}

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
"""

    response = llm.invoke(prompt)

    return response.content, source_documents


# ---------------------------------------------------------
# 7. Streamlit Main
# ---------------------------------------------------------
def main():

    st.set_page_config(
        page_title="Streamlit RAG",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Private Data Q/A Chat")

    st.info(
        "PDF, DOCX, PPTX 문서를 업로드한 뒤 "
        "Process 버튼을 눌러주세요."
    )

    # -----------------------------------------------------
    # Session State
    # -----------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------
    with st.sidebar:

        st.header("Upload your file")

        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=[
                "pdf",
                "docx",
                "pptx",
            ],
            accept_multiple_files=True,
        )

        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            key="chatbot_api_key",
        )

        process = st.button(
            "Process",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Process 버튼
    # -----------------------------------------------------
    if process:

        if not google_api_key:

            st.warning(
                "Google API Key를 입력해 주세요."
            )

            st.stop()

        if not uploaded_files:

            st.warning(
                "먼저 문서를 업로드해 주세요."
            )

            st.stop()

        os.environ["GOOGLE_API_KEY"] = google_api_key

        with st.spinner(
            "문서를 읽고 벡터DB를 생성하고 있습니다..."
        ):

            # 문서 읽기
            docs = get_text(uploaded_files)

            if len(docs) == 0:

                st.error(
                    "문서에서 읽을 수 있는 텍스트가 없습니다."
                )

                st.stop()

            # Chunk 생성
            text_chunks = get_text_chunks(docs)

            # FAISS 생성
            vectordb = get_vectorstore(
                text_chunks
            )

            st.session_state.vectordb = vectordb
            st.session_state.processComplete = True

        st.success(
            f"처리 완료! "
            f"{len(docs)}개 문서 페이지/요소를 읽고 "
            f"{len(text_chunks)}개의 Chunk를 생성했습니다."
        )

    # -----------------------------------------------------
    # 최초 AI 메시지
    # -----------------------------------------------------
    if len(st.session_state.messages) == 0:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content":
                    "안녕하세요! "
                    "문서를 업로드하고 Process 버튼을 누른 뒤 "
                    "문서에 대해 질문해 주세요."
            }
        )

    # -----------------------------------------------------
    # 이전 대화 출력
    # -----------------------------------------------------
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # -----------------------------------------------------
    # 사용자 질문
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

        with st.chat_message("user"):

            st.markdown(query)

        # 아직 Process를 하지 않은 경우
        if not st.session_state.processComplete:

            with st.chat_message(
                "assistant"
            ):

                st.warning(
                    "먼저 문서를 업로드하고 "
                    "Process 버튼을 눌러주세요."
                )

        else:

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        response, source_documents = ask_gemini(
                            query,
                            st.session_state.vectordb,
                            google_api_key,
                        )

                        st.markdown(response)

                        # -------------------------------
                        # 참고 문서 출력
                        # 교수님 PDF 85~89라인 역할
                        # -------------------------------
                        with st.expander(
                            "📑 참고 문서 확인"
                        ):

                            for i, doc in enumerate(
                                source_documents
                            ):

                                source = doc.metadata.get(
                                    "source",
                                    "알 수 없음",
                                )

                                page = doc.metadata.get(
                                    "page",
                                    None,
                                )

                                st.markdown(
                                    f"### 참고 문서 {i + 1}"
                                )

                                st.write(
                                    f"출처: {source}"
                                )

                                if page is not None:

                                    st.write(
                                        f"페이지: {page + 1}"
                                    )

                                st.write(
                                    doc.page_content
                                )

                                st.divider()

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                            }
                        )

                    except Exception as e:

                        st.error(
                            f"오류가 발생했습니다: {e}"
                        )


# ---------------------------------------------------------
# 프로그램 시작
# ---------------------------------------------------------
if __name__ == "__main__":
    main()