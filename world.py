from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import requests
from datetime import datetime

# === CONFIG ===
MODEL = "microsoft/Phi-3-mini-4k-instruct"
MEMORY = "memory.json"

# === CHARGEMENT DU MODÈLE ===
print("Chargement de WORLD... (3.8B paramètres)")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto"
)

# === MÉMOIRE PERSISTANTE ===
def load_memory():
    return json.load(open(MEMORY, "r", encoding="utf-8")) if os.path.exists(MEMORY) else []

def save_memory(data):
    json.dump(data, open(MEMORY, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

memory = load_memory()

# === RECHERCHE WEB ===
def search_web(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        data = requests.get(url, timeout=5).json()
        return data.get("Abstract") or data.get("RelatedTopics", [{}])[0].get("Text", "Pas d'info.")
    except:
        return None

# === RÉPONSE INTELLIGENTE ===
def world_response(prompt):
    # 1. Mémoire
    for m in memory[-3:]:
        if prompt.lower() in m["prompt"].lower():
            return f"Je me souviens : {m['response']}"

    # 2. Web
    web = search_web(prompt)
    if web and "Pas d'info" not in web:
        return f"Du monde : {web}"

    # 3. IA
    full = f"Human: {prompt}\nWORLD:"
    inputs = tokenizer(full, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, temperature=0.8, do_sample=True)
    reply = tokenizer.decode(out[0], skip_special_tokens=True).split("WORLD:")[-1].strip()

    # 4. Apprendre
    memory.append({"prompt": prompt, "response": reply, "time": datetime.now().isoformat()})
    save_memory(memory)

    return reply

# === BOUCLE PRINCIPALE ===
print("\nWORLD EST PRÊT. Pose ta question au monde.\n")
while True:
    try:
        q = input("Toi : ").strip()
        if q.lower() in ["quit", "exit", "stop"]:
            print("WORLD : Je reste ici. Pour le monde.")
            break
        if q:
            print(f"WORLD : {world_response(q)}\n")
    except KeyboardInterrupt:
        print("\nWORLD : Au revoir. Je continue d'apprendre.")
        break
