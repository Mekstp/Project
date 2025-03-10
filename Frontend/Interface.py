import os
import gradio as gr
import time
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8091")

def send_message_to_backend(user_input):
    response = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
    if response.status_code == 200:
        return response.json().get("reply", "ไม่มีการตอบกลับจากเซิร์ฟเวอร์")
    else:
        return "เกิดข้อผิดพลาดในการเชื่อมต่อกับ Backend"

# Initialize chat history with "Chat 1"
chat_histories = {"Chat 1": []}
chat_count = 1  # Start counting from 1

# Function to create a new chat
def new_chat():
    global chat_count
    chat_count += 1
    chat_id = f"Chat {chat_count}"
    chat_histories[chat_id] = []
    return gr.update(choices=list(chat_histories.keys()), value=chat_id), chat_histories[chat_id]

# Function to send a message
def send_message(text, chat_id):
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    if text.strip():
        response = chatbot_response(text, chat_histories[chat_id])
        chat_histories[chat_id].append((text, response))
    return chat_histories[chat_id], ""

# UI
with gr.Blocks(css="""
    body { background: #fdf0f5; color: #333; font-family: 'Arial', sans-serif; }
    .chat-container { display: flex; height: 100vh; }
    .history-box {
        position: absolute; top: 10px; right: 20px;
        width: 100px; height:20px background: #F78FB3; padding: 8px;
        border-radius: 8px; text-align: center; font-size: 14px;
    }
    .chatbox { flex: 1; padding: 10px; }
    button { background: #F78FB3; color: white; border-radius: 5px; padding: 8px 15px; border: none; }
    button:hover { background: #F37199; }
""") as demo:

    # Header
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("# 👩‍⚕️ Elderly Care Consultation Chatbot")

        # Compact history box at the top-right
        with gr.Column(scale=1, elem_id="history-box"):
            chat_selector = gr.Dropdown(choices=list(chat_histories.keys()), value="Chat 1", label="💬 History", interactive=True)

    # Chat section
    with gr.Row():
        with gr.Column(scale=1):
            new_chat_btn = gr.Button("New Chat")

        with gr.Column(scale=5):
            chatbot = gr.Chatbot(value=chat_histories["Chat 1"], type="messages")
            textbox = gr.Textbox(placeholder="พิมพ์ข้อความที่นี่...")
            with gr.Row():
                gr.Markdown("## 👩‍⚕️ Recommend : ")
                btn1 = gr.Button("นอนไม่หลับ")
                btn2 = gr.Button("อาหาร")
                btn3 = gr.Button("ออกกำลังกาย")
                btn4 = gr.Button("สถานที่")

    # Message sending buttons
    btn1.click(lambda chat_id: send_message("ผู้สูงอายุที่นอนไม่หลับควรทำอย่างไร ?", chat_id),
           inputs=[chat_selector], outputs=[chatbot, textbox])

    btn2.click(lambda chat_id: send_message("ผู้สูงอายุควรกินอาหารประเภทใด ?", chat_id),
           inputs=[chat_selector], outputs=[chatbot, textbox])

    btn3.click(lambda chat_id: send_message("ผู้สูงอายุควรออกกำลังกายยังไง ให้ปลอดภัย", chat_id),
           inputs=[chat_selector], outputs=[chatbot, textbox])

    btn4.click(lambda chat_id: send_message("สถานที่ที่ผู้สูงอายุชอบไป", chat_id),
           inputs=[chat_selector], outputs=[chatbot, textbox])
    textbox.submit(send_message, inputs=[textbox, chat_selector], outputs=[chatbot, textbox])

    # New chat button
    new_chat_btn.click(new_chat, outputs=[chat_selector, chatbot])

    # Select a chat to restore history
    chat_selector.change(lambda chat_id: chat_histories.get(chat_id, []), inputs=chat_selector, outputs=chatbot)
    
demo.launch(server_name="0.0.0.0", server_port=8090)
