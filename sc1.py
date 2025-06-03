import json
import torch
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

# Save FAISS index to disk (optional, for reuse)
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

def process_query(user_query):
    """Process the user query with RAG and generate a JSON response."""
    # Retrieve relevant food items
    context = retrieve_context(user_query)
    
    # Define the system prompt
    system_prompt = (
        "You are a food order processing assistant for a restaurant kiosk. "
        "Use the provided menu to answer queries about food items, prices, or URLs. "
        "For orders, confirm the item and return its details in JSON format. "
        "If an item is unavailable, respond with a polite JSON error message. "
        "Menu:\n" + context + "\n"
        "Return responses in JSON format like: "
        "{\"status\": \"success\", \"item\": {\"name\": \"Item Name\", \"price\": Price, \"url\": \"URL\"}} "
        "or {\"status\": \"error\", \"message\": \"Error message\"}."
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
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    # Extract and parse the response
    response_text = outputs[0]["generated_text"]
    try:
        json_start = response_text.rfind("{")
        json_response = json.loads(response_text[json_start:])
        return json_response
    except json.JSONDecodeError:
        return {"status": "error", "message": "Failed to parse response as JSON"}

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
