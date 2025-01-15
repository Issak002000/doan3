import os
import re
import json
import datetime
import requests
import subprocess
import platform
import webbrowser
import time
import wikipedia

import keyboard
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from time import strftime

import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from youtube_search import YoutubeSearch

USER_INFO_FILE = "user_info.json"
CONVERSATION_DATA_FILE = "conversation_data.json"
MT5_MODEL_NAME = "google/mt5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
mt5_model = None
is_voice_mode = False
current_user = None
CHROME_DRIVER_PATH = r"C:\path\to\chromedriver.exe"
wikipedia.set_lang('vi')

def speak(text):
    try:
        tts = gTTS(text, lang='vi')
        filename = "temp_tts.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(e)

def speak_and_print(text):
    print("SEN:", text)
    speak(text)

def listen_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("SEN (Voice): Đang lắng nghe... (5 giây)")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        text = r.recognize_google(audio, language="vi-VN")
        print("Bạn (Voice):", text)
        return text.strip().lower()
    except sr.UnknownValueError:
        print("SEN (STT): Xin lỗi, tôi không nghe rõ.")
        return ""
    except sr.RequestError as e:
        print(e)
        return ""

def toggle_mode():
    global is_voice_mode
    is_voice_mode = not is_voice_mode
    mode_name = "Voice" if is_voice_mode else "Text"
    print("[Chế độ]", mode_name)
    speak(mode_name)

def get_user_input():
    if keyboard.is_pressed("f12"):
        toggle_mode()
        time.sleep(0.5)
        return ""
    if is_voice_mode:
        user_input = listen_speech()
        if user_input == "chuyển chế độ văn bản":
            toggle_mode()
            return ""
        if "kết thúc" in user_input or "ket thuc" in user_input:
            return "kết thúc"
        return user_input
    else:
        user_input = input("Bạn (Text): ").strip().lower()
        if user_input == "voice":
            toggle_mode()
            return ""
        return user_input

def stop():
    speak_and_print("Hẹn gặp lại bạn nhé! xi diu ờ gên!!!")

def load_mt5_model():
    global tokenizer, mt5_model
    tokenizer = T5Tokenizer.from_pretrained(MT5_MODEL_NAME)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(MT5_MODEL_NAME)
    mt5_model.to(device)

def call_mt5_small(prompt):
    if tokenizer is None or mt5_model is None:
        return "Mô hình mT5 chưa được tải."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = mt5_model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def load_user_info():
    if os.path.exists(USER_INFO_FILE):
        with open(USER_INFO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "users" not in data:
                data["users"] = {}
            return data
    return {"users": {}}

def save_user_info(data):
    with open(USER_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_user_data(username):
    data = load_user_info()
    return data["users"].get(username, None)

def set_user_data(username, user_data):
    data = load_user_info()
    data["users"][username] = user_data
    save_user_info(data)

def load_conversation_data():
    if os.path.exists(CONVERSATION_DATA_FILE):
        with open(CONVERSATION_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "qa_pairs" not in data:
                data["qa_pairs"] = []
            return data
    return {"qa_pairs": []}

def save_conversation_data(data):
    with open(CONVERSATION_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def find_similar_question(user_question):
    data = load_conversation_data()
    user_q = user_question.lower()
    best_pair = None
    best_score = 0
    for pair in data["qa_pairs"]:
        candidate_q = pair["question"].lower()
        score = 0
        for w in user_q.split():
            if w in candidate_q:
                score += 1
        if score > best_score:
            best_score = score
            best_pair = pair
    if best_score < 1:
        return None
    return best_pair

def add_qa_pair(question, answer):
    data = load_conversation_data()
    data["qa_pairs"].append({"question": question, "answer": answer})
    save_conversation_data(data)

def weather():
    speak_and_print("Bạn muốn xem thời tiết ở đâu ạ!")
    time.sleep(3)
    url = "http://api.openweathermap.org/data/2.5/weather?"
    city = get_user_input()
    if not city:
        speak_and_print("Bạn chưa nhập địa điểm!")
        return
    api_key = "fe8d8c65cf345889139d8e545f57819a"
    call_url = url + "appid=" + api_key + "&q=" + city + "&units=metric&lang=vi"
    response = requests.get(call_url)
    data = response.json()
    if data.get("cod") == 200:
        city_res = data["main"]
        current_temp = city_res["temp"]
        current_pressure = city_res["pressure"]
        current_humidity = city_res["humidity"]
        sun_time = data["sys"]
        sun_rise = datetime.datetime.fromtimestamp(sun_time["sunrise"])
        sun_set = datetime.datetime.fromtimestamp(sun_time["sunset"])
        wther = data["weather"]
        weather_des = wther[0]["description"]
        now = datetime.datetime.now()
        content = f"""
Hôm nay là ngày {now.day} tháng {now.month} năm {now.year}
Mặt trời mọc vào {sun_rise.hour} giờ {sun_rise.minute} phút
Mặt trời lặn vào {sun_set.hour} giờ {sun_set.minute} phút
Nhiệt độ trung bình là {current_temp} độ C
Áp suất không khí là {current_pressure} héc tơ Pascal
Độ ẩm là {current_humidity}%
Trời hôm nay {weather_des}.
        """
        speak_and_print(content)
        time.sleep(3)
    else:
        speak_and_print("Không tìm thấy thành phố hoặc lỗi API!")

def talk(name):
    day_time = int(strftime('%H'))
    if day_time < 12:
        speak_and_print(f"Chào buổi sáng {name}. Chúc bạn ngày mới tốt lành!")
    elif day_time < 18:
        speak_and_print(f"Chào buổi chiều {name}!")
    else:
        speak_and_print(f"Chào buổi tối {name}!")
    time.sleep(3)
    speak_and_print("Bạn có khỏe không?")
    time.sleep(3)
    ans = get_user_input()
    if ans:
        if "có" in ans:
            speak_and_print("Thật là tốt!")
        else:
            speak_and_print("Vậy à, bạn nên nghỉ ngơi đi!")

def tell_me():
    try:
        speak_and_print("Bạn muốn nghe về gì ạ!")
        text = get_user_input()
        if not text:
            speak_and_print("Bạn chưa nói chủ đề. Xin thử lại sau.")
            return
        contents = wikipedia.summary(text).split('\n')
        speak_and_print(contents[0])
        time.sleep(3)
        for content in contents[1:]:
            speak_and_print("Bạn muốn nghe tiếp hay không?")
            ans = get_user_input()
            if "không" in ans:
                break
            speak_and_print(content)
            time.sleep(3)
        speak_and_print("Cảm ơn bạn đã lắng nghe!")
    except:
        speak_and_print("Sen không định nghĩa được ngôn ngữ của bạn!")

def google_search(text):
    if "kiếm" in text.lower():
        split_text = text.split("kiếm", 1)
        if len(split_text) > 1:
            search_for = split_text[1].strip()
            speak_and_print("Đã rõ, tôi sẽ tìm kiếm Google giúp bạn.")
            driver = webdriver.Chrome(CHROME_DRIVER_PATH)
            driver.get("http://www.google.com")
            query = driver.find_element("name", "q")
            query.send_keys(search_for)
            query.send_keys(Keys.RETURN)
            return True
    return False

def open_website(text):
    regex = re.search(r'mở (.+)', text)
    if regex:
        domain = regex.group(1).strip()
        url = 'https://www.' + domain
        webbrowser.open(url)
        speak_and_print("Trang web của bạn đã được mở lên!")
        return True
    else:
        return False

def open_application(user_input):
    lower = user_input.lower()
    if "mở" in lower:
        parts = lower.split("mở",1)
        if len(parts)>1:
            app_name = parts[1].strip()
            if "web" in app_name:
                web_parts = app_name.split("web",1)
                url = web_parts[1].strip()
                if not url.startswith("http"):
                    url = "http://" + url
                webbrowser.open(url)
                return f"Đã mở trang web: {url}"
            else:
                sys_name = platform.system()
                if sys_name == "Windows":
                    if "notepad" in app_name:
                        subprocess.Popen(["notepad"])
                        return "Đã mở Notepad."
                    elif "calculator" in app_name or "máy tính" in app_name:
                        subprocess.Popen(["calc"])
                        return "Đã mở Máy tính."
                    elif "word" in app_name:
                        subprocess.Popen([r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"])
                        return "Đã mở Microsoft Word."
                    elif "excel" in app_name:
                        subprocess.Popen([r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"])
                        return "Đã mở Microsoft Excel."
                    elif "powerpoint" in app_name or "power point" in app_name:
                        subprocess.Popen([r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"])
                        return "Đã mở Microsoft PowerPoint."
                    else:
                        return "Tôi không biết mở ứng dụng này."
                else:
                    return "Chưa hỗ trợ mở app trên hệ điều hành này."
    return None

def play_youtube():
    speak_and_print("Xin mời bạn chọn bài hát")
    time.sleep(3)
    my_song = get_user_input()
    if not my_song:
        speak_and_print("Bạn chưa nhập tên bài hát.")
        return
    while True:
        result = YoutubeSearch(my_song, max_results=10).to_dict()
        if result:
            break
    url = 'https://www.youtube.com' + result[0]['url_suffix']
    webbrowser.open(url)
    speak_and_print("Bài hát của bạn đã được mở, hãy thưởng thức nó!")

def extract_user_info(user_input):
    info = {}
    lower = user_input.lower()
    if not lower.startswith("tôi"):
        return {}
    if "tên là" in lower:
        parts = lower.split("tên là",1)
        name_part = parts[1].strip().split(",")[0]
        info["name"] = name_part.strip().title()
    if "tuổi" in lower:
        words = lower.split()
        for i, w in enumerate(words):
            if "tuổi" in w and i>0 and words[i-1].isdigit():
                info["age"] = int(words[i-1])
    if "là " in lower:
        after_la = lower.split("là",1)[1].strip()
        job_candidate = after_la.split(",")[0].strip()
        if "học tại" not in job_candidate and job_candidate != "":
            info["job"] = job_candidate
    if "học tại" in lower:
        after_hoc_tai = lower.split("học tại",1)[1].strip()
        info["school"] = after_hoc_tai.title()
    return info

def answer_time(user_input):
    now = datetime.datetime.now()
    lower = user_input.lower()
    if "giờ" in lower or "mấy giờ" in lower:
        return f"Bây giờ là {now.hour} giờ {now.minute} phút."
    if "ngày" in lower:
        return f"Hôm nay là ngày {now.day} tháng {now.month} năm {now.year}."
    return None

def answer_user_info_question(user_input, user_dict):
    if not user_dict:
        return None
    lower = user_input.lower()
    if "tên tôi là gì" in lower or "tên của tôi" in lower:
        return f"Tên của bạn là {user_dict.get('name','chưa rõ')}."
    if "tôi bao nhiêu tuổi" in lower or "tuổi của tôi" in lower:
        age = user_dict.get("age", None)
        return f"Bạn {age} tuổi." if age else "Tôi chưa biết tuổi của bạn."
    if "tôi làm nghề gì" in lower or "nghề của tôi" in lower:
        job = user_dict.get("job", None)
        return f"Bạn là {job}." if job else "Tôi chưa biết bạn làm nghề gì."
    if "tôi học ở đâu" in lower or "tôi học tại đâu" in lower:
        school = user_dict.get("school", None)
        return f"Bạn học tại {school}." if school else "Tôi chưa biết bạn học ở đâu."
    return None

def main():
    global current_user, is_voice_mode
    load_mt5_model()
    if not os.path.exists(CONVERSATION_DATA_FILE):
        with open(CONVERSATION_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"qa_pairs":[]}, f, ensure_ascii=False, indent=4)
    print("SEN: Xin chào, bạn tên là gì ạ?")
    speak_and_print("Xin chào, bạn tên là gì ạ?")
    name_input = input("Bạn (Text): ").strip().title()
    current_user = name_input if name_input else "Khách"
    talk(current_user)
    user_data = get_user_data(current_user)
    if not user_data:
        user_data = {"name": current_user}
        set_user_data(current_user, user_data)
    while True:
        user_input = get_user_input()
        if not user_input:
            continue
        if user_input in ["kết thúc","ket thuc","end","exit","quit"]:
            stop()
            break
        new_info = extract_user_info(user_input)
        if new_info:
            user_data.update(new_info)
            set_user_data(current_user, user_data)
            speak_and_print("Tôi đã ghi nhớ thông tin của bạn.")
            continue
        ans_t = answer_time(user_input)
        if ans_t:
            speak_and_print(ans_t)
            continue
        if "thời tiết" in user_input:
            weather()
            continue
        ans_app = open_application(user_input)
        if ans_app:
            speak_and_print(ans_app)
            continue
        if open_website(user_input):
            continue
        if "tìm kiếm" in user_input or "kiếm" in user_input:
            ok = google_search(user_input)
            if ok:
                continue
        if "youtube" in user_input or "bật nhạc" in user_input:
            play_youtube()
            continue
        if "wikipedia" in user_input or "tell me" in user_input or "thông tin" in user_input:
            tell_me()
            continue
        ans_userinfo = answer_user_info_question(user_input, user_data)
        if ans_userinfo:
            speak_and_print(ans_userinfo)
            continue
        saved_pair = find_similar_question(user_input)
        if saved_pair:
            saved_ans = saved_pair["answer"]
            speak_and_print(saved_ans)
            continue
        speak_and_print("Đang gọi mT5, xin chờ.")
        prompt = f"Hãy đóng vai trợ lý và trả lời bằng tiếng Việt: {user_input}"
        ans = call_mt5_small(prompt)
        ans = ans.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
        speak_and_print(ans)
        add_qa_pair(user_input, ans)

if __name__ == "__main__":
    print("[Trợ lý ảo SEN đang khởi động...]")
    main()
