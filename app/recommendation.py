# recommendation.py
import os
import torch
# import torchvision
from datasets import load_from_disk
# from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever
from transformers import AutoTokenizer, RagTokenForGeneration, RagRetriever

DATASET_PATH = "data/my_custom_dataset"
INDEX_PATH = os.path.join(DATASET_PATH, "embeddings")
MODEL_NAME = "facebook/rag-token-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du dataset + index
dataset_hf = load_from_disk(DATASET_PATH)
if not dataset_hf.is_index_initialized("embeddings"):
    dataset_hf.load_faiss_index("embeddings", INDEX_PATH)

# Chargement du modèle RAG
# tokenizer = RagTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = RagTokenForGeneration.from_pretrained(MODEL_NAME, strict=False)
retriever = RagRetriever.from_pretrained(
    MODEL_NAME,
    index_name="custom",
    passages_path=DATASET_PATH,
    index_path=INDEX_PATH
)
model.to(device)

def recommend_movies(user_query: str, top_n=5):
    if not user_query.strip():
        return []

    # 1) Encoder la requête
    inputs = tokenizer(user_query, return_tensors="pt").to(device)
    with torch.no_grad():
        question_hidden_states = model.question_encoder(input_ids=inputs["input_ids"])[0]
    hidden_states_np = question_hidden_states.cpu().numpy()

    # 2) Récupérer les doc_ids avec le retriever
    docs_dict = retriever(
        question_hidden_states=hidden_states_np,
        question_input_ids=inputs["input_ids"],
        n_docs=top_n
    )
    doc_ids = docs_dict["doc_ids"][0]

    # 3) Récupération "à la main" (sans .select())
    retrieved_rows = []
    for doc_id in doc_ids:
        doc_id = int(doc_id)  # s’assurer que c'est un entier Python
        row = dataset_hf[doc_id]  # indexation directe
        retrieved_rows.append(row)

    # 4) Construire la liste de recommandations
    recommendations = []
    for row in retrieved_rows:
        recommendations.append({
            "title": row.get("title", "Untitled"),
            "overview": row.get("overview", ""),
            "poster_path": row.get("poster_path", None)
        })

    return recommendations
