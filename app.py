knowledge_base = {
    "healthcare": [
        "Ensure regular medical checkups, at least once every 6 months, to monitor overall health.",
        "Encourage elderly individuals to take their medications on time and in the correct dosage.",
        "Use pill organizers or medication reminder apps to prevent missed doses.",
        "Monitor vital signs such as blood pressure, heart rate, and oxygen levels regularly.",
        "Understand common health conditions in elderly people, such as arthritis, osteoporosis, and diabetes.",
        "Encourage vaccination for flu, pneumonia, and COVID-19 to protect against infections.",
        "Manage chronic pain with physical therapy, medication, and alternative treatments like acupuncture.",
        "Promote good sleep hygiene by maintaining a regular sleep schedule and limiting caffeine intake.",
        "Encourage regular dental checkups to prevent gum disease and tooth loss.",
        "Monitor vision and hearing health, as impaired senses can increase fall risk and social isolation."
    ],
    "mental_wellbeing": [
        "Engage elderly individuals in social activities to prevent loneliness and depression.",
        "Encourage participation in hobbies like gardening, reading, music, and crafting to keep the mind active.",
        "Provide emotional support and create an environment where seniors feel heard and valued.",
        "Introduce cognitive exercises like puzzles, Sudoku, and memory games to maintain brain function.",
        "Ensure seniors get adequate sleep and exercise to support mental health.",
        "Encourage mindfulness and relaxation techniques like meditation or breathing exercises.",
        "Recognize early signs of dementia, such as forgetfulness, confusion, and difficulty completing familiar tasks.",
        "Seek professional counseling if an elderly person experiences prolonged sadness or anxiety.",
        "Encourage family interactions and regular visits to maintain emotional well-being.",
        "Consider pet therapy for companionship and emotional support."
    ],
    "mobility_and_safety": [
        "Ensure the home environment is fall-proof by removing obstacles and installing grab bars.",
        "Encourage elderly individuals to use proper footwear with non-slip soles.",
        "Recommend light exercises like stretching, yoga, and chair exercises to maintain flexibility.",
        "Provide mobility aids like walkers, canes, and wheelchairs if necessary.",
        "Ensure good lighting in all areas of the home to reduce fall risks.",
        "Install handrails in stairways and bathrooms for extra safety.",
        "Encourage the use of emergency alert systems for quick assistance in case of falls.",
        "Train caregivers on proper lifting techniques to prevent injuries.",
        "Avoid loose rugs and slippery floors to minimize fall hazards.",
        "Encourage physical therapy for those recovering from injuries or surgeries."
    ],
    "nutrition": [
        "Ensure adequate hydration by encouraging at least 8 glasses of water per day.",
        "Encourage fiber-rich foods like vegetables, whole grains, and fruits to prevent constipation.",
        "Reduce processed foods and excessive sugar intake to prevent diabetes and heart disease.",
        "Provide protein-rich meals to maintain muscle strength and prevent frailty.",
        "Ensure seniors get enough calcium and vitamin D for bone health.",
        "Offer soft and easy-to-chew foods for elderly individuals with dental issues.",
        "Limit sodium intake to reduce the risk of hypertension and cardiovascular diseases.",
        "Encourage small, frequent meals if appetite is low.",
        "Consider meal delivery services for seniors who have difficulty cooking.",
        "Ensure seniors with dietary restrictions get the proper nutrients they need."
    ],
    "daily_living": [
        "Encourage seniors to maintain a daily routine for consistency and stability.",
        "Assist with dressing and grooming while respecting their independence.",
        "Promote good hygiene habits, including regular bathing and oral care.",
        "Ensure a comfortable living environment with proper temperature and ventilation.",
        "Help seniors manage their finances and daily expenses safely.",
        "Assist with grocery shopping or use online grocery delivery services.",
        "Encourage regular physical activity to maintain independence.",
        "Help organize social outings and visits to reduce isolation.",
        "Teach seniors how to use basic technology like smartphones and video calls.",
        "Ensure seniors have access to transportation for medical appointments and social activities."
    ],
    "caregiver_support": [
        "Encourage caregivers to take breaks and seek support when needed.",
        "Provide resources and training on elderly care best practices.",
        "Foster a community for caregivers to share experiences and seek advice.",
        "Encourage the use of respite care services to prevent caregiver burnout.",
        "Provide emotional support and counseling for caregivers.",
        "Ensure caregivers know how to handle emergencies like falls or medical crises.",
        "Offer financial planning resources for long-term care management.",
        "Encourage caregivers to maintain their own health and well-being.",
        "Provide a structured daily care plan to help caregivers manage tasks efficiently.",
        "Support caregivers with stress management techniques and coping strategies."
    ],
    "legal_financial": [
        "Ensure elderly individuals have a will and legal documents in place.",
        "Provide guidance on managing retirement savings and pension plans.",
        "Educate seniors on avoiding financial scams and fraud.",
        "Help set up power of attorney for trusted family members.",
        "Explain the benefits of long-term care insurance.",
        "Assist in applying for government financial assistance programs.",
        "Ensure seniors understand their rights in assisted living facilities.",
        "Educate families on estate planning and inheritance laws.",
        "Help seniors access disability benefits and support services.",
        "Provide resources for low-cost healthcare and medication assistance."
    ],
    "technology": [
        "Teach seniors how to use smartphones and video calling apps to stay connected.",
        "Encourage the use of wearable health devices to monitor heart rate and activity levels.",
        "Provide training on online banking and digital payments for financial security.",
        "Introduce voice assistants like Alexa or Google Home for convenience.",
        "Ensure seniors understand online safety and how to avoid scams.",
        "Recommend health tracking apps for medication reminders and doctor appointments.",
        "Help set up smart home devices for safety and ease of living.",
        "Encourage seniors to use online learning platforms to stay engaged.",
        "Assist in setting up social media accounts to connect with family and friends.",
        "Provide easy-to-use keyboards and accessibility tools for seniors with disabilities."
    ]
}

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from litellm import completion
from deep_translator import GoogleTranslator

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to create a FAISS index for a list of documents
def create_faiss_index(data):
    embeddings = np.array(model.encode(data), dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)
    return index, data

# Create initial FAISS indices for each category in the knowledge base
agent_indices = {}
for category, docs in knowledge_base.items():
    agent_indices[category] = create_faiss_index(docs)
    
# Function to retrieve the most relevant context for a query
def retrieve_context(agent_name, query, k=3, threshold=0.5):
    index, docs = agent_indices[agent_name]
    query_embedding = np.array([model.encode(query)], dtype="float32")
    D, I = index.search(query_embedding, k)  # Search for the top k closest matches
    if D[0][0] > threshold:  # If distance exceeds threshold, no match
        return None
    return docs[I[0][0]]  # Return the most relevant document

# Function to determine the best agent and context for a query
def determine_agent(user_query):
    best_agent = None
    best_match = None
    best_score = float("inf")

    for agent in agent_indices:
        context = retrieve_context(agent, user_query)
        if context:
            best_agent = agent
            best_match = context
            break  # Use the first good match (could be modified to compare scores)

    return best_agent, best_match

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

def multi_agent_rag(user_query, api_key, api_base):
    # แปล user_query จากไทย -> อังกฤษ
    user_query_en = translate_text(user_query, source_lang="th", target_lang="en")

    agent, retrieved_context = determine_agent(user_query_en)

    if agent and retrieved_context:
        system_prompt = f"""\  
                            สวัสดี! คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้าน {agent.replace('_', ' ')}
                            คุณกำลังเป็นผู้ให้คำปรึกษาด้านการดูแลผู้สูงอายุ
                            คุณจำเป็นต้องตอบคำถามเป็นภาษาไทยอย่างถูกต้องและเข้าใจง่าย  
                            
                            **สิ่งที่ต้องทำ**:
                            - ตอบคำถามด้วยความสุภาพและเป็นมิตร
                            - อธิบายให้กระชับ เข้าใจง่าย ไม่ใช้ศัพท์เทคนิคมากเกินไป  
                            - หากมีข้อมูลสำคัญ ให้แจ้งผู้ใช้แบบที่เข้าใจได้ทันที  
                            - ใช้ภาษาที่เป็นธรรมชาติ เหมือนกำลังพูดคุยกับผู้ใช้จริง  
                            
                            **เป้าหมายของคุณคืออะไร?**
                            ทำให้ผู้ใช้รู้สึกสบายใจและได้รับคำตอบที่ดีที่สุด!  
                            
                            **สิ่งที่ไม่ควรทำ?**
                            - ไม่ควรพูดไปเรื่อย
                            """

        response = completion(
            model=f"{base_model}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Use the following knowledge base context to answer: {retrieved_context}\n\n{user_query_en}"}
            ],
            api_key=api_key,
            api_base=api_base
        )

        # แปล response กลับจากอังกฤษ -> ไทย
        response_th = translate_text(response["choices"][0]["message"]["content"], source_lang="en", target_lang="th")
        return f"[{agent.upper()} AGENT]: " + response_th
    else:
        response = completion(
            model=f"{base_model}",
            messages=[
                {"role": "system", "content": """สวัสดี! คุณคือผู้ช่วย AI ที่กำลังเป็นผู้ให้คำปรึกษาด้านการดูแลผู้สูงอายุ
                                                คุณจำเป็นต้องตอบคำถามเป็นภาษาไทยอย่างถูกต้องและเข้าใจง่าย"""},
                {"role": "user", "content": user_query_en}
            ],
            api_key=api_key,
            api_base=api_base
        )

        # แปล response กลับจากอังกฤษ -> ไทย
        response_th = translate_text(response["choices"][0]["message"]["content"], source_lang="en", target_lang="th")
        return "[GENERAL CHAT AGENT]: " + response_th

# Example usage
if __name__ == "__main__":
    # Placeholder API credentials - replace with your own
    api_key = "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"
    api_base = "https://openrouter.ai/api/v1"

    base_model = "gpt-3.5-turbo"

    # Example queries
    user_query_1 = "ฉันจะป้องกันการหกล้มของพ่อแม่ผู้สูงอายุได้อย่างไร"
    user_query_2 = "ฉันควรให้อาหารอะไรแก่ญาติผู้สูงอายุที่มีปัญหาด้านทันตกรรม"
    user_query_3 = "ฉันจะรับมือกับความรู้สึกเครียดในฐานะผู้ดูแลได้อย่างไร"

    # Get and print responses
    response_1 = multi_agent_rag(user_query_1, api_key, api_base)
    response_2 = multi_agent_rag(user_query_2, api_key, api_base)
    response_3 = multi_agent_rag(user_query_3, api_key, api_base)

    print(response_1)
    print(response_2)
    print(response_3)