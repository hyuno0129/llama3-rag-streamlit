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
# 1. Gemini 최신 응답에서 텍스트만 추출
# ---------------------------------------------------------
def extract_response_text(content):

    # 예전 Gemini/LangChain 응답
    if isinstance(content, str):
        return content

    # 최신 Gemini content block 형식
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

    return str(content)


# ---------------------------------------------------------
# 2. 토큰 길이 계산
# ---------------------------------------------------------
def tiktoken_len(text):

    tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = tokenizer.encode(text)

    return len(tokens)


# ---------------------------------------------------------
# 3. 업로드된 문서 읽기
# PDF / DOCX / PPTX
# ---------------------------------------------------------
def get_text(docs):

    doc_list = []

    for doc in docs:

        file_name = doc.name

        suffix = os.path.splitext(file_name)[1]

        # Streamlit 업로드 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(doc.getvalue())

            temp_path = temp_file.name

        logger.info(f"문서 처리 시작: {file_name}")

        try:

            # PDF
            if file_name.lower().endswith(".pdf"):

                loader = PyPDFLoader(temp_path)

                documents = loader.load()

            # Word
            elif file_name.lower().endswith(".docx"):

                loader = Docx2txtLoader(temp_path)

                documents = loader.load()

            # PowerPoint
            elif file_name.lower().endswith(".pptx"):

                loader = UnstructuredPowerPointLoader(temp_path)

                documents = loader.load()

            else:

                continue

            # 실제 파일명을 source에 저장
            for document in documents:

                document.metadata["source"] = file_name

            doc_list.extend(documents)

        finally:

            try:
                os.remove(temp_path)

            except Exception:
                pass

    return doc_list


# ---------------------------------------------------------
# 4. 문서 Chunk 분할
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
# 5. FAISS VectorStore 생성
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
# 6. 검색된 문서를 Gemini용 Context 문자열로 변환
# ---------------------------------------------------------
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
            page_text = f"페이지: {page + 1}\n"

        context += f"""

[참고 문서 {i + 1}]
출처: {source}
{page_text}

{doc.page_content}

"""

    return context


# ---------------------------------------------------------
# 7. Gemini 일반 질문
# ---------------------------------------------------------
def ask_general_gemini(question, google_api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    response = llm.invoke(question)

    answer = extract_response_text(
        response.content
    )

    return answer


# ---------------------------------------------------------
# 8. Gemini + RAG 질문
# ---------------------------------------------------------
def ask_gemini(question, vectordb, google_api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=google_api_key,
    )

    # 질문과 가장 관련된 chunk 5개 검색
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        },
    )

    source_documents = retriever.invoke(
        question
    )

    context = format_documents(
        source_documents
    )

    prompt = f"""
당신은 업로드된 문서를 분석하여 질문에 답하는 RAG 챗봇입니다.

아래의 [참고 문서]를 자세히 확인한 뒤
사용자의 질문에 답하세요.

반드시 다음 규칙을 지키세요.

1. 먼저 참고 문서 안에서 질문과 관련된 내용을 찾으세요.
2. 관련 정보가 있으면 그 내용을 근거로 정확하게 답하세요.
3. 여러 참고 문서에 정보가 나뉘어 있으면 내용을 종합해서 답하세요.
4. 참고 문서에 정말 관련 정보가 없는 경우에만
   "업로드된 문서에서 해당 정보를 찾을 수 없습니다."
   라고 답하세요.
5. 문서에 없는 사실을 임의로 만들어내지 마세요.
6. 질문과 직접 관련된 내용을 우선해서 설명하세요.
7. 한국어로 자연스럽고 이해하기 쉽게 답하세요.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
"""

    response = llm.invoke(prompt)

    answer = extract_response_text(
        response.content
    )

    return answer, source_documents


# ---------------------------------------------------------
# 9. Streamlit Main
# ---------------------------------------------------------
def main():

    st.set_page_config(
        page_title="Streamlit RAG",
        page_icon="📚",
        layout="wide",
    )

    st.title(
        "📚 Private Data Q/A Chat"
    )

    st.info(
        "PDF, DOCX, PPTX 문서를 업로드한 뒤 "
        "Process 버튼을 눌러주세요. "
        "문서를 처리하지 않은 상태에서는 "
        "일반 Gemini 질문도 가능합니다."
    )


    # -----------------------------------------------------
    # Session State 초기화
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

        st.header(
            "Upload your file"
        )

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
    # Process 버튼 클릭 시
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


        os.environ[
            "GOOGLE_API_KEY"
        ] = google_api_key


        with st.spinner(
            "문서를 읽고 벡터DB를 생성하고 있습니다..."
        ):

            # 문서 읽기
            docs = get_text(
                uploaded_files
            )


            if len(docs) == 0:

                st.error(
                    "문서에서 읽을 수 있는 "
                    "텍스트가 없습니다."
                )

                st.stop()


            # Chunk 생성
            text_chunks = get_text_chunks(
                docs
            )


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
    if len(
        st.session_state.messages
    ) == 0:

        st.session_state.messages.append(
            {
                "role": "assistant",

                "content":
                    "안녕하세요! "
                    "일반 질문을 바로 하거나, "
                    "문서를 업로드하고 Process 버튼을 누른 뒤 "
                    "문서 내용에 대해 질문해 주세요."
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
    # 사용자 질문 입력
    # -----------------------------------------------------
    query = st.chat_input(
        "질문을 입력해 주세요."
    )


    if query:

        # 사용자 메시지 저장
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )


        with st.chat_message("user"):

            st.markdown(query)


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    # -------------------------------------
                    # 문서가 Process 된 경우
                    # RAG 질문
                    # -------------------------------------
                    if st.session_state.processComplete:

                        answer, source_documents = ask_gemini(
                            query,
                            st.session_state.vectordb,
                            google_api_key,
                        )

                        st.markdown(
                            answer
                        )


                        # ---------------------------------
                        # 참고 문서 표시
                        # ---------------------------------
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


                    # -------------------------------------
                    # 문서를 Process하지 않은 경우
                    # Gemini 일반 질문
                    # -------------------------------------
                    else:

                        answer = ask_general_gemini(
                            query,
                            google_api_key,
                        )

                        st.markdown(
                            answer
                        )


                    # AI 답변 저장
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


# ---------------------------------------------------------
# 프로그램 실행
# ---------------------------------------------------------
if __name__ == "__main__":

    main()
