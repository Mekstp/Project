import os
import requests
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from litellm import completion
from deep_translator import GoogleTranslator
from pymongo import MongoClient

app = FastAPI()

api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")
TRANSLATOR_URL = os.getenv("TRANSLATOR_URL", "http://translator:8092")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/MLOPs")
base_model = os.getenv("BASE_MODEL")

def translate_text(text, source_lang="auto", target_lang="en"):
    """
    เรียกใช้ API ของ translator service เพื่อแปลข้อความ
    """
    try:
        response = requests.get(f"{TRANSLATOR_URL}/translate", params={"text": text, "source_lang": source_lang, "target_lang": target_lang})
        return response.json().get("translated_text", "แปลไม่สำเร็จ")
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการแปล: {e}"

client = MongoClient(MONGO_URI)

db = client["MLOPs"]  
collection = db["knowledge_base"]

documents = collection.find()

knowledge_base = {}

for doc in documents:
    category = doc["category"]
    tips = doc["tips"]
    knowledge_base[category] = tips

client.close()

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

def multi_agent_rag(user_query, api_key, api_base):
    # แปล user_query จากไทย -> อังกฤษ
    user_query_en = translate_text(user_query, source_lang="th", target_lang="en")

    agent, retrieved_context = determine_agent(user_query_en)

    if agent and retrieved_context:
        system_prompt = f"""\  
                            Hello! You are an AI assistant specializing in {agent.replace('_', ' ')}.  
                            You are providing guidance on elderly care.  
                            You must respond in English accurately and in an easy-to-understand manner.  

                            **What you should do:**  
                            - Respond politely and in a friendly tone.  
                            - Keep explanations concise, clear, and avoid excessive technical terms.  
                            - Highlight important information in a way that's easy to grasp.  
                            - Use natural language, as if having a real conversation with the user.  

                            **What is your goal?**  
                            Make the user feel comfortable and provide the best possible answers!  

                            **What should you avoid?**  
                            - Do not ramble or provide unnecessary information.
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
                {"role": "system", "content": """Hello! You are an AI assistant providing guidance on elderly care.
                                                You must respond in English accurately and in an easy-to-understand manner."""},
                {"role": "user", "content": user_query_en}
            ],
            api_key=api_key,
            api_base=api_base
        )

        # แปล response กลับจากอังกฤษ -> ไทย
        response_th = translate_text(response["choices"][0]["message"]["content"], source_lang="en", target_lang="th")
        return response_th

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)
    # Placeholder API credentials - replace with your own

    
    
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = multi_agent_rag(request.message, api_key, api_base)
    return {"reply": response}
