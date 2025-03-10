from fastapi import FastAPI
from deep_translator import GoogleTranslator

app = FastAPI()

def translate_text(text, source_lang="auto", target_lang="en"):
    """
    แปลข้อความจากภาษาต้นทางไปยังภาษาปลายทางโดยใช้ GoogleTranslator จาก deep-translator

    :param text: ข้อความที่ต้องการแปล
    :param source_lang: ภาษาต้นทาง (ค่าเริ่มต้นเป็น "auto" ให้ระบบตรวจจับอัตโนมัติ)
    :param target_lang: ภาษาปลายทาง (ค่าเริ่มต้นเป็น "en" - อังกฤษ)
    :return: ข้อความที่ถูกแปล
    """
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการแปล: {e}"

@app.get("/")
def read_root():
    return {"message": "Translator service is running!"}

@app.get("/translate")
def translate_endpoint(text: str, source_lang: str = "auto", target_lang: str = "en"):
    return {"translated_text": translate_text(text, source_lang, target_lang)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)
