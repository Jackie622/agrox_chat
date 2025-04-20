import io
import time
import wave
import requests
import speech_recognition as sr
from tqdm import tqdm
import re
from gradio_client import Client, handle_file
from openai import OpenAI
import os

class AudioRecorder:
    def __init__(self, rate=16000):
        """初始化录音器，设置采样率"""
        self.rate = rate
        self.recognizer = sr.Recognizer()

    def record(self):
        """录制音频并返回音频数据"""
        with sr.Microphone(sample_rate=self.rate) as source:
            print("请在倒计时结束前说话", flush=True)
            time.sleep(0.1)  # 确保print输出
            start_time = time.time()
            audio = None

            for _ in tqdm(range(20), desc="倒计时", unit="s"):
                try:
                    # 录音，设置超时1秒以便更新进度条
                    audio = self.recognizer.listen(
                        source, timeout=1, phrase_time_limit=15
                    )
                    break  # 录音成功，跳出循环
                except sr.WaitTimeoutError:
                    # 超时未检测到语音
                    if time.time() - start_time > 20:
                        print("未检测到语音输入")
                        break

            if audio is None:
                print("未检测到语音输入")
                return None

        # 返回音频数据
        audio_data = audio.get_wav_data()
        return io.BytesIO(audio_data)

    def save_wav(self, audio_data, filename="temp_output.wav"):
        """将音频数据保存为WAV文件"""
        audio_data.seek(0)
        with wave.open(filename, "wb") as wav_file:
            nchannels = 1
            sampwidth = 2  # 16-bit audio
            framerate = self.rate  # 采样率
            comptype = "NONE"
            compname = "not compressed"
            audio_frames = audio_data.read()

            wav_file.setnchannels(nchannels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            wav_file.setcomptype(comptype, compname)
            wav_file.writeframes(audio_frames)
        audio_data.seek(0)

    def run(self):
        """运行录音功能并保存音频文件"""
        audio_data = self.record()
        filename = "temp_output.wav"
        if audio_data:
            self.save_wav(audio_data, filename)
        return filename


class SenseVoice:
    def __init__(self, api_url, emo=False):
        """初始化语音识别接口，设置API URL和情感识别开关"""
        self.api_url = api_url
        self.emo = emo

    def _extract_second_bracket_content(self, raw_text):
        """提取文本中第二对尖括号内的内容"""
        match = re.search(r"<[^<>]*><([^<>]*)>", raw_text)
        if match:
            return match.group(1)
        return None

    def _get_speech_text(self, audio_data):
        """将音频数据发送到API并获取识别结果"""
        print("正在进行语音识别")
        files = [("files", audio_data)]
        data = {"keys": "audio1", "lang": "auto"}

        response = requests.post(self.api_url, files=files, data=data)
        if response.status_code == 200:
            result_json = response.json()
            if "result" in result_json and len(result_json["result"]) > 0:
                if self.emo:
                    result = (
                        self._extract_second_bracket_content(
                            result_json["result"][0]["raw_text"]
                        )
                        + "\n"
                        + result_json["result"][0]["text"]
                    )
                    return result
                else:
                    return result_json["result"][0]["text"]
            else:
                return "未识别到有效的文本"
        else:
            return f"请求失败，状态码: {response.status_code}"

    def speech_to_text(self, audio_data):
        """调用API进行语音识别并返回结果"""
        return self._get_speech_text(audio_data)


# 使用示例
if __name__ == "__main__":
    recorder = AudioRecorder()
    audio_data = recorder.run()

    if audio_data:
        client = Client("http://127.0.0.1:7860/")
        inputText = client.predict(
            input_wav=handle_file(audio_data),
            language="zh",
            api_name="/model_inference",
        )
        print(inputText)

        aiClient = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )

        response = aiClient.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个精通百科知识的知识官。"},
                {"role": "user", "content": inputText},
            ],
            stream=False,
        )

        print(response.choices[0].message.content)

