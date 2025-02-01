#feedback_analysis.py
import os
import pandas as pd
from datasets import load_from_disk
import faiss
import numpy as np

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Chemins vers les fichiers et dossier de données
FEEDBACK_FILE = "data/feedback.csv"
DATASET_PATH = "data/my_custom_dataset"
INDEX_PATH = os.path.join(DATASET_PATH, "embeddings")

# Paramètres pour les ajustements granulaires
PENALTY_RATE = 0.05  # Facteur maximal de pénalisation (exemple : si 100% des mots-clés sont trouvés, multiplier par 0.95)
BOOST_RATE = 0.05    # Facteur maximal de boost (exemple : si 100% des mots-clés sont trouvés, multiplier par 1.05)

# Initialisation de KeyBERT avec le modèle SentenceTransformer
kw_extractor = KeyBERT(model=SentenceTransformer("all-mpnet-base-v2"))

def extract_keywords(text, top_n=3):
    """
    Extrait les mots-clés principaux d'un texte.
    Retourne une liste de mots-clés (chaînes de caractères).
    """
    if not text.strip():
        return []
    keywords = kw_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    # Retourne uniquement les mots-clés sans les scores
    return [kw[0] for kw in keywords]

def adjust_embeddings_based_on_feedback():
    """
    Lit le fichier feedback.csv pour récupérer à la fois les feedbacks négatifs et positifs,
    extrait des mots-clés (depuis le commentaire ou le titre en absence de commentaire),
    et ajuste la norme des embeddings FAISS pour les films contenant ces mots-clés.
    
    Pour chaque film du dataset, un multiplicateur est calculé de manière cumulative :
      - Les feedbacks négatifs pénalisent l'embedding (multiplier par un facteur entre 1 et 0.95).
      - Les feedbacks positifs boostent l'embedding (multiplier par un facteur entre 1 et 1.05).
    
    Ensuite, l'index FAISS est reconstruit avec ces embeddings ajustés.
    """
    if not os.path.exists(FEEDBACK_FILE):
        print("Aucun feedback. Le fichier feedback.csv n'existe pas.")
        return

    # Chargement du feedback
    df_feedback = pd.read_csv(FEEDBACK_FILE)
    if df_feedback.empty:
        print("Aucun feedback trouvé.")
        return

    # Normaliser la colonne "liked"
    df_feedback["liked"] = df_feedback["liked"].astype(str).str.strip().str.lower()

    # Séparation des feedbacks négatifs et positifs
    df_neg = df_feedback[df_feedback["liked"] == "false"]
    df_pos = df_feedback[df_feedback["liked"] == "true"]

    # Chargement du dataset et de l'index FAISS existant
    dataset = load_from_disk(DATASET_PATH)
    dataset.load_faiss_index("embeddings", INDEX_PATH)
    index = dataset.get_index("embeddings").faiss_index
    nb_vectors = index.ntotal
    d = index.d

    # Reconstruction de tous les embeddings dans un tableau numpy
    all_embs = np.zeros((nb_vectors, d), dtype=np.float32)
    for i in range(nb_vectors):
        all_embs[i] = index.reconstruct(i)

    # Récupérer le texte associé à chaque film
    # On privilégie la colonne "text" (par exemple une concaténation de title+overview) si elle existe
    if "text" in dataset.column_names:
        all_texts = dataset["text"][:]
    else:
        all_texts = dataset["overview"][:]

    # Initialiser un tableau de multiplicateurs (un par film), initialement à 1 (aucun ajustement)
    multipliers = np.ones(nb_vectors, dtype=np.float32)

    def count_keyword_occurrences(text, keywords):
        """
        Retourne le nombre de mots-clés présents dans le texte (comparaison insensible à la casse).
        """
        text_lower = text.lower()
        count = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                count += 1
        return count

    # Traitement des feedbacks négatifs : pénalisation
    for _, row in df_neg.iterrows():
        # Assurer que 'comment' est une chaîne de caractères
        comment = row.get("comment", "")
        if not isinstance(comment, str):
            comment = ""
        # Extraire les mots-clés depuis le commentaire ; si vide, utiliser le titre
        keywords = extract_keywords(comment, top_n=3) if comment.strip() else []
        if not keywords:
            title = row.get("title", "")
            if not isinstance(title, str):
                title = ""
            if title.strip():
                keywords = [title]
        if not keywords:
            continue  # Aucun mot-clé à exploiter pour ce feedback

        # Pour chaque film, calculer le nombre de mots-clés trouvés
        for i in range(nb_vectors):
            count = count_keyword_occurrences(all_texts[i], keywords)
            if count > 0:
                proportion = count / len(keywords)  # proportion de correspondance
                # Le facteur de pénalisation varie de 1 (aucune pénalisation) à (1 - PENALTY_RATE) (correspondance totale)
                penalty_factor = 1 - (proportion * PENALTY_RATE)
                multipliers[i] *= penalty_factor

    # Traitement des feedbacks positifs : boost
    for _, row in df_pos.iterrows():
        comment = row.get("comment", "")
        if not isinstance(comment, str):
            comment = ""
        # Pour un feedback positif, si un commentaire est fourni, en extraire des mots-clés,
        # sinon utiliser le titre comme indicateur.
        keywords = extract_keywords(comment, top_n=3) if comment.strip() else []
        if not keywords:
            title = row.get("title", "")
            if not isinstance(title, str):
                title = ""
            if title.strip():
                keywords = [title]
        if not keywords:
            continue

        # Pour chaque film, calculer la correspondance avec les mots-clés positifs
        for i in range(nb_vectors):
            count = count_keyword_occurrences(all_texts[i], keywords)
            if count > 0:
                proportion = count / len(keywords)
                # Le facteur de boost varie de 1 (aucun boost) à (1 + BOOST_RATE) (correspondance totale)
                boost_factor = 1 + (proportion * BOOST_RATE)
                multipliers[i] *= boost_factor

    # Appliquer le multiplicateur à chaque embedding
    for i in range(nb_vectors):
        all_embs[i] *= multipliers[i]

    # Reconstruire un nouvel index FAISS avec les embeddings ajustés
    new_index = faiss.IndexFlatIP(d)  # Utilisation de la similarité par produit intérieur (IP)
    new_index.add(all_embs)
    faiss.write_index(new_index, INDEX_PATH)

    print("Ré-indexation terminée !")
    print(f"Nombre de feedbacks négatifs traités : {len(df_neg)}")
    print(f"Nombre de feedbacks positifs traités : {len(df_pos)}")

if __name__ == "__main__":
    adjust_embeddings_based_on_feedback()
