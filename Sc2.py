import json
import torch
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="TinyLlama Food Order API")

# Load food item dataset
with open("food_items.json", "r") as f:
    food_items = json.load(f)

# Initialize embedding model for RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for food items
item_texts = [f"{item['name']} - Price: ${item['price']} - URL: {item['url']}" for item in food_items]
item_embeddings = embedder.encode(item_texts, convert_to_numpy=True)

# Initialize FAISS index
dimension = item_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(item_embeddings)

# Save FAISS index to disk
faiss.write_index(index, "food_items_index.faiss")

# Initialize TinyLlama pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define request model for the API
class OrderRequest(BaseModel):
    query: str

def retrieve_context(query, k=2):
    """Retrieve top-k relevant food items using FAISS."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    context = []
    for idx in indices[0]:
        if idx < len(food_items):
            item = food_items[idx]
            context.append(f"Name: {item['name']}, Price: ${item['price']}, URL: {item['url']}")
    return "\n".join(context) if context else "No relevant items found."

def clean_json_response(response_text):
    """Extract and clean JSON from model output."""
    # Find the last JSON-like structure
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return None
    json_str = json_match.group(0)
    # Remove any trailing/leading non-JSON text
    json_str = json_str.strip()
    # Fix common JSON issues (e.g., unescaped quotes)
    json_str = re.sub(r'(?<!\\)"', '\\"', json_str, count=1)  # Escape inner quotes if needed
    return json_str

def process_query(user_query):
    """Process the user query with RAG and generate a JSON response."""
    # Retrieve relevant food items
    context = retrieve_context(user_query)
    
    # Define the system prompt (strengthened for JSON-only output)
    system_prompt = (
        "You are a food order processing assistant for a restaurant kiosk. "
        "Use the provided menu to answer queries about food items, prices, or URLs. "
        "For orders, confirm the item and return its details in JSON format. "
        "If an item is unavailable, respond with a polite JSON error message. "
        "Menu:\n" + context + "\n"
        "Return ONLY valid JSON, with no extra text, comments, or explanations. "
        "Examples:\n"
        "{\"status\": \"success\", \"item\": {\"name\": \"Item Name\", \"price\": 12.99, \"url\": \"URL\"}}\n"
        "{\"status\": \"error\", \"message\": \"Sorry, that item is not available.\"}"
    )

    # Define the conversation messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    # Apply the chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate a response
    outputs = pipe(
        prompt,
        max_new_tokens=150,  # Reduced to limit extraneous output
        do_sample=True,
        temperature=0.6,  # Lowered for more deterministic output
        top_k=40,        # Adjusted for better coherence
        top_p=0.9
    )

    # Extract and clean the response
    response_text = outputs[0]["generated_text"]
    json_str = clean_json_response(response_text)
    if not json_str:
        return {"status": "error", "message": "No valid JSON found in model response"}
    
    try:
        json_response = json.loads(json_str)
        return json_response
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Failed to parse response as JSON: {str(e)}"}

@app.post("/order")
async def handle_order(request: OrderRequest):
    """API endpoint to handle food order queries."""
    try:
        response = process_query(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the server on 0.0.0.0 to allow LAN access
    uvicorn.run(app, host="0.0.0.0", port=8000)
