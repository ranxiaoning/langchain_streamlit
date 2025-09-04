# langchain_streamlit# RAG文档问答系统
<img width="3189" height="1644" alt="image" src="https://github.com/user-attachments/assets/9ae4a21b-4308-4e95-891a-820e73664427" />
<img width="3538" height="1635" alt="image" src="https://github.com/user-attachments/assets/97621f7c-7337-4d8a-a4f9-3e7d69ff0782" />

一个基于Streamlit、LangChain和自定义OpenAI API构建的检索增强生成（RAG）文档问答系统。该系统允许用户上传PDF或文本文件，然后基于文档内容进行智能问答。

## 功能特性

- 📄 **多格式文档支持**：支持PDF和TXT格式文档上传
- 🔍 **智能检索**：使用FAISS向量数据库进行高效的语义检索
- 🤖 **AI问答**：基于GPT模型生成准确、上下文相关的答案
- 🌊 **流式输出**：实时流式显示AI生成内容，提升用户体验
- 📱 **友好界面**：简洁直观的Web界面，易于使用
- 🔒 **安全可靠**：支持自定义API端点，保障数据安全

## 技术栈

- **前端框架**: Streamlit
- **语言模型**: OpenAI GPT系列 (支持gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- **向量数据库**: FAISS
- **文本处理**: LangChain
- **文档解析**: PyPDFLoader, TextLoader

## 安装步骤

### 前提条件

- Python 3.8+
- pip包管理工具
- OpenAI API密钥

### 安装依赖

1. 克隆或下载本项目文件
2. 安装所需依赖：

```bash
pip install streamlit langchain-community faiss-cpu openai pypdf
```

或者使用requirements.txt安装：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动应用

```bash
streamlit run app.py
```

### 2. 配置API设置

在左侧边栏中：
- 输入您的OpenAI API密钥
- 设置API基础URL（默认为 `https://ai.nengyongai.cn/v1`）
- 选择要使用的模型

### 3. 上传文档

- 点击"上传文档"按钮，选择PDF或TXT文件
- 支持多文件同时上传

### 4. 处理文档

- 点击"处理文档"按钮，系统将解析和索引文档内容
- 处理完成后会显示加载的文本片段数量

### 5. 开始问答

- 在聊天输入框中输入您的问题
- 系统将从文档中检索相关信息并生成答案
- 可以点击"查看来源文档"查看答案的依据来源



## 核心组件说明

### 自定义嵌入类 (CustomOpenAIEmbeddings)

使用OpenAI的text-embedding-3-small模型为文档和查询生成嵌入向量。

### 自定义LLM类 (CustomOpenAILLM)

封装OpenAI聊天完成API，提供统一的LLM调用接口。

### 流式处理器 (StreamHandler)

实现实时流式输出，增强用户体验。

### 自定义检索QA类 (CustomRetrievalQA)

结合检索器和LLM，实现完整的RAG问答流程。

## 配置选项

### 模型选择

系统支持以下OpenAI模型：
- gpt-4o-mini (默认)
- gpt-4o
- gpt-3.5-turbo

### 文本分割参数

- 块大小 (chunk_size): 1000字符
- 重叠大小 (chunk_overlap): 200字符



### Q: 如何提高答案质量？

A: 可以尝试：
1. 上传更相关、更高质量的文档
2. 调整检索参数k值
3. 使用更强大的模型（如gpt-4o）


## 支持

如有问题或建议，请通过GitHub Issues提交反馈。

---

**注意**: 使用本系统需要自行提供OpenAI API密钥，并承担相应的API使用费用。请妥善保管您的API密钥，不要泄露给他人。
