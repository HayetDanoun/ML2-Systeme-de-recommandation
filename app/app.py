#app.py
import streamlit as st
import pandas as pd
import random
import os

from recommendation import recommend_movies
from feedback_handler import store_feedback
from feedback_analysis import adjust_embeddings_based_on_feedback  

# =========================================
# ========== Configuration Streamlit ======
# =========================================
st.set_page_config(page_title="Recommandation de Films", layout="wide")

DATA_DIR = "data"
MOVIE_CSV = os.path.join(DATA_DIR, "movies_metadata_clean.csv")

def load_initial_recommendations(df_path=MOVIE_CSV, top_n=50):
    if not os.path.exists(df_path):
        return []
    df_all = pd.read_csv(df_path)
    df_all = df_all.dropna(subset=["poster_path", "vote_average"])
    best_movies = df_all.sort_values(by="vote_average", ascending=False).head(top_n)
    return best_movies.to_dict(orient="records")

# =========================================
# ========== Variables de session =========
# =========================================
if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False

if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

if "initial_recommendations" not in st.session_state:
    st.session_state.initial_recommendations = load_initial_recommendations()

if "feedback_inputs" not in st.session_state:
    st.session_state.feedback_inputs = {}

if "reindex_done" not in st.session_state:
    st.session_state.reindex_done = False

if st.session_state.reindex_done:
    st.success("Merci, votre avis sera pris en compte")
    st.session_state.reindex_done = False

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>🎬 Système de Recommandation de Films</h1>
        <p style='font-size: 1.2rem; color: #555;'>
            Décrivez le film que vous souhaitez voir, et nous vous proposerons des recommandations !
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    user_input = st.text_area(
        "Décrivez votre film idéal :",
        placeholder="Example : an action movie with superheroes and humor"
    )
    top_n = st.slider("Nombre de recommandations :", 1, 10, 5)
    if st.button("Obtenir des recommandations"):
        if user_input.strip():
            recs = recommend_movies(user_input, top_n=top_n)
            st.session_state.recommendations = pd.DataFrame(recs)
            st.session_state.show_recommendations = True
        else:
            st.warning("Veuillez entrer une description pour obtenir des recommandations.")

with col2:
    if st.session_state.show_recommendations and not st.session_state.recommendations.empty:
        st.subheader(f"🎥 Voici les {top_n} films proches de votre description :")
        df = st.session_state.recommendations
    else:
        st.subheader("🎥 Les films les mieux notés :")
        df = pd.DataFrame(
            random.sample(
                st.session_state.initial_recommendations,
                min(len(st.session_state.initial_recommendations), top_n)
            )
        )

    for idx, row in df.iterrows():
        movie_title = row.get('title', 'Untitled')
        st.markdown(f"**{movie_title}** ")
        st.write(row.get('overview', 'N/A'))

        if st.session_state.show_recommendations:
            feedback_key = f"feedback_{idx}"
            comment_key = f"comment_{idx}"

            feedback_choice = st.radio(
                f"Votre avis sur **{movie_title}** ?",
                ("Aucun", "J’aime 👍", "Je n’aime pas 👎"),
                key=feedback_key,
                horizontal=True
            )
            st.session_state.feedback_inputs[feedback_key] = feedback_choice

            if feedback_choice == "Je n’aime pas 👎":
                comment = st.text_area(
                    f"Pourquoi vous n’aimez pas « {movie_title} » ?",
                    key=comment_key
                )
                st.session_state.feedback_inputs[comment_key] = comment
            else:
                st.session_state.feedback_inputs[comment_key] = ""

        st.write("---")

    # Le bouton d'application n'apparaît que si l'utilisateur a demandé des recommandations
    if st.session_state.show_recommendations:
        if st.button("Appliquer la ré-indexation maintenant"):
            for idx, row in st.session_state.recommendations.iterrows():
                feedback_key = f"feedback_{idx}"
                comment_key = f"comment_{idx}"
                feedback_choice = st.session_state.feedback_inputs.get(feedback_key, "Aucun")
                user_comment = st.session_state.feedback_inputs.get(comment_key, "")
                
                if feedback_choice == "J’aime 👍":
                    store_feedback(
                        user_query=user_input,
                        title=row.get("title", None),
                        liked=True,
                        comment="",
                        mode="per_movie"
                    )
                elif feedback_choice == "Je n’aime pas 👎":
                    store_feedback(
                        user_query=user_input,
                        title=row.get("title", None),
                        liked=False,
                        comment=user_comment,
                        mode="per_movie"
                    )
            # Appel à la fonction qui ajuste les embeddings en fonction des feedbacks
            adjust_embeddings_based_on_feedback()
            st.session_state.reindex_done = True
            st.rerun()
            st.success("Ré-indexation terminée ! Les ajustements (pénalisation/boost) ont été appliqués.")

st.markdown(
    """
    <div style='margin-top: 2rem; text-align: center; font-size: 0.9rem; color: #777;'>
        🔍 Système de recommandation de films - Data Science & Machine Learning 🎥
    </div>
    """,
    unsafe_allow_html=True
)
