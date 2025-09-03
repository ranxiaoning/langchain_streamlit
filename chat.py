import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from io import BytesIO
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from typing import List, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import threading
from typing import Dict, List, Any, Optional
import json


# 自定义 Embeddings 类，使用自定义的 OpenAI 客户端
class CustomOpenAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
        self.model = "text-embedding-3-small"  # 可以根据需要更改模型

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"生成嵌入向量时出错: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入向量"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"生成查询嵌入向量时出错: {e}")
            raise


# 自定义 LLM 类，使用自定义的 OpenAI 客户端
class CustomOpenAILLM:
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def __call__(self, prompt: str, stop: List[str] = None) -> str:
        """调用模型生成回复"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"调用模型时出错: {e}")
            return f"错误: {str(e)}"


# 流式输出处理器
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.lock = threading.Lock()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        with self.lock:
            self.text += token
            self.container.markdown(self.text + "▌")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        with self.lock:
            self.container.markdown(self.text)


# 设置页面标题和布局
st.set_page_config(
    page_title="RAG文档问答系统",
    page_icon="📚",
    layout="wide"
)

# 标题和说明
st.title("📚 RAG文档问答系统")
st.markdown("上传您的文档，然后基于文档内容提问。系统会从文档中检索相关信息并生成答案。")

# 初始化session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "client" not in st.session_state:
    st.session_state.client = None

# 侧边栏 - API密钥设置和文档上传
with st.sidebar:
    st.header("设置")

    # API密钥和基础URL输入
    api_key = st.text_input("API密钥", type="password", value="")
    base_url = st.text_input("API基础URL", value="https://ai.nengyongai.cn/v1")
    model_name = st.selectbox("选择模型", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])

    # 初始化客户端
    if api_key and base_url:
        try:
            st.session_state.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            st.success("API客户端初始化成功！")
        except Exception as e:
            st.error(f"初始化API客户端失败: {e}")

    # 文档上传
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # 处理按钮
    process_btn = st.button("处理文档")

# 处理文档
if process_btn and uploaded_files and st.session_state.client:
    with st.spinner("正在处理文档..."):
        try:
            documents = []

            # 读取上传的文件
            for uploaded_file in uploaded_files:
                file_bytes = BytesIO(uploaded_file.read())

                if uploaded_file.type == "application/pdf":
                    # 保存临时PDF文件
                    with open("temp.pdf", "wb") as f:
                        f.write(file_bytes.getvalue())
                    loader = PyPDFLoader("temp.pdf")
                    docs = loader.load()
                    documents.extend(docs)
                    os.remove("temp.pdf")

                elif uploaded_file.type == "text/plain":
                    # 处理文本文件
                    text = str(file_bytes.read(), "utf-8")
                    # 使用文件内容创建TextLoader
                    with open("temp.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    loader = TextLoader("temp.txt")
                    docs = loader.load()
                    documents.extend(docs)
                    os.remove("temp.txt")

            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # 创建自定义的embeddings
            embeddings = CustomOpenAIEmbeddings(st.session_state.client)

            # 创建向量存储
            vector_store = FAISS.from_documents(splits, embeddings)

            # 创建自定义的LLM
            llm = CustomOpenAILLM(st.session_state.client, model=model_name)

            # 创建QA链
            prompt_template = """使用以下上下文片段来回答最后的问题。
            如果你不知道答案，就说你不知道，不要编造答案。
            尽量使答案详细且全面。

            {context}

            问题: {question}
            答案:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )


            # 由于我们使用自定义LLM，需要手动实现检索和生成逻辑
            class CustomRetrievalQA:
                def __init__(self, retriever, llm, prompt_template):
                    self.retriever = retriever
                    self.llm = llm
                    self.prompt_template = prompt_template

                def __call__(self, query):
                    # 检索相关文档
                    relevant_docs = self.retriever.get_relevant_documents(query)

                    # 构建上下文
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # 构建提示词
                    prompt = self.prompt_template.format(context=context, question=query)

                    # 调用LLM生成答案
                    answer = self.llm(prompt)

                    return {
                        "result": answer,
                        "source_documents": relevant_docs
                    }


            # 创建自定义的QA链
            qa_chain = CustomRetrievalQA(
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                llm=llm,
                prompt_template=PROMPT
            )

            # 保存到session state
            st.session_state.vector_store = vector_store
            st.session_state.qa_chain = qa_chain

            st.success(f"文档处理完成！已加载 {len(splits)} 个文本片段。")

        except Exception as e:
            st.error(f"处理文档时出错: {str(e)}")
elif process_btn:
    if not st.session_state.client:
        st.error("请先配置API密钥和基础URL")
    elif not uploaded_files:
        st.error("请上传至少一个文档")

# 聊天界面
if st.session_state.qa_chain:
    # 显示聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("查看来源文档"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**来源 {i + 1}:** {source.page_content[:200]}...")

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("思考中..."):
                try:
                    result = st.session_state.qa_chain(prompt)

                    # 流式输出
                    response_text = ""
                    for char in result["result"]:
                        response_text += char
                        message_placeholder.markdown(response_text + "▌")

                    message_placeholder.markdown(response_text)

                    # 显示来源文档（如果有）
                    if result["source_documents"]:
                        with st.expander("查看来源文档"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**来源 {i + 1}:** {doc.page_content[:200]}...")

                    # 添加助手消息到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": result["source_documents"]
                    })

                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")
else:
    st.info("👈 请在侧边栏配置API信息并上传文档以开始对话")

# 页脚
st.markdown("---")
st.markdown("基于自定义OpenAI API和LangChain构建的RAG文档问答系统")