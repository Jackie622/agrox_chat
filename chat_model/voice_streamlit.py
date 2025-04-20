import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
import io
import time
import wave
import requests
import re
from tqdm import tqdm
from gradio_client import Client, handle_file
from PIL import Image
import speech_recognition as sr
import threading

# 将父目录放入系统路径中，可根据实际路径调整
# sys.path.append(r'D:\My_Files\实验室学习相关\大模型部署')

# ---------------------------------
# 语音录制模块（含倒计时提示）
# ---------------------------------
class AudioRecorder:
    def __init__(self, rate=16000, timeout_seconds=20):
        self.rate = rate
        self.timeout_seconds = timeout_seconds
        self.recognizer = sr.Recognizer()

    def record(self):
        with sr.Microphone(sample_rate=self.rate) as source:
            st.info(f"请在倒计时结束前说话（最长 {self.timeout_seconds} 秒）")
            # UI 占位
            countdown_text = st.empty()
            progress_bar = st.progress(0)

            # 后台线程负责实际录音
            audio_container = {}
            def _listen():
                try:
                    audio_container["audio"] = self.recognizer.listen(
                        source,
                        timeout=None,                     # 等待首句，不超时
                        phrase_time_limit=self.timeout_seconds  # 最长录制时间
                    )
                except Exception:
                    pass

            t = threading.Thread(target=_listen, daemon=True)
            t.start()

            # 前台更新倒计时与进度
            for elapsed in range(self.timeout_seconds):
                if not t.is_alive():
                    break
                remaining = self.timeout_seconds - elapsed
                countdown_text.info(f"倒计时剩余：{remaining} 秒")
                progress_bar.progress((elapsed + 1) / self.timeout_seconds)
                time.sleep(1)

            # 清理 UI
            countdown_text.empty()
            progress_bar.empty()

            # 判断是否有录音
            audio_data = audio_container.get("audio", None)
            if audio_data is None:
                st.warning("未检测到语音输入或录音超时")
                return None

            return audio_data.get_wav_data()

    def save_wav(self, wav_bytes: bytes, filename="temp_output.wav"):
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.rate)
            wav_file.writeframes(wav_bytes)

    def run(self):
        wav_bytes = self.record()
        if wav_bytes:
            fname = "temp_output.wav"
            self.save_wav(wav_bytes, fname)
            return fname
        return None

    def save_wav(self, wav_bytes: bytes, filename="temp_output.wav"):
        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.rate)
            wav_file.writeframes(wav_bytes)

    def run(self):
        wav_bytes = self.record()
        if wav_bytes:
            fname = "temp_output.wav"
            self.save_wav(wav_bytes, fname)
            return fname
        return None


# ---------------------------------
# 语音识别器，优先 Gradio，失败后提示
# ---------------------------------
class SpeechRecognizer:
    def __init__(self, gradio_url="http://127.0.0.1:7860/"):
        try:
            self.client = Client(gradio_url)
        except Exception as e:
            self.client = None
            st.warning(f"无法连接 Gradio ASR 服务: {e}")

    def recognize(self, file_path: str):
        if not self.client:
            st.error("ASR 服务未连接，请启动 Gradio 服务或配置 ASR API。")
            return None
        try:
            result = self.client.predict(
                input_wav=handle_file(file_path),
                language="zh",
                api_name="/model_inference"
            )
            return result
        except Exception as e:
            st.error(f"语音识别失败: {e}")
            return None

# ---------------------------------
# LangChain 检索问答模块
# ---------------------------------
def get_retriever():
    from langchain_community.embeddings import ZhipuAIEmbeddings
    from langchain_chroma import Chroma
    embedding = ZhipuAIEmbeddings(model="embedding-3", dimensions=2048)
    persist_directory = r'chroma'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb.as_retriever()


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        temperature=1.0
    )
    condense_sys = (
        "请根据聊天记录总结用户最近的问题，" \
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_prompt = ChatPromptTemplate([
        ("system", condense_sys),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
        condense_prompt | llm | StrOutputParser() | retriever,
    )
    system_prompt = (
        "你是一个农场助手。请使用检索到的上下文片段回答这个问题。"
        "请使用简洁的话语回答用户。\n\n{context}"  # 如果你不知道答案就说不知道。
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough().assign(context=retrieve_docs).assign(answer=qa_chain)
    return qa_history_chain


def gen_response(chain, user_input, chat_history):
    for res in chain.stream({"input": user_input, "chat_history": chat_history}):
        if "answer" in res:
            yield res["answer"]

# ---------------------------------
# Streamlit 应用
# ---------------------------------
def main():
    st.markdown("""
    <style>
        .stImage {text-align: center !important;}
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(Image.open("白.png"), width=200)
    st.markdown('### 🧑‍🌾知耘农业大模型')

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_qa_history_chain()

    recorder = AudioRecorder()
    recognizer = SpeechRecognizer()
    messages = st.container()

    for role, text in st.session_state.messages:
        with messages.chat_message(role):
            st.write(text)

    col_txt, col_voice = st.columns([4,1])
    user_input = None
    if prompt := col_txt.chat_input("输入问题或使用语音…"):
        user_input = prompt
    if col_voice.button("🎙️ 语音输入"):
        wav_path = recorder.run()
        if wav_path:
            st.info("正在识别语音…")
            text = recognizer.recognize(wav_path)
            if text:
                user_input = text
                st.info(f"识别结果：{text}")
        else:
            st.warning("录音失败，请重试")

    if user_input:
        st.session_state.messages.append(("human", user_input))
        with messages.chat_message("human"): st.write(user_input)
        answer_stream = gen_response(st.session_state.qa_chain, user_input, st.session_state.messages)
        with messages.chat_message("ai"):
            ai_resp = st.write_stream(answer_stream)
        st.session_state.messages.append(("ai", ai_resp))

if __name__ == "__main__":
    main()
