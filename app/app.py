import streamlit as st
import pandas as pd
from recommendation import get_movie_recommendations, dataset_hf

# ------------------------
# Interface utilisateur Streamlit
# ------------------------
st.set_page_config(page_title="Recommandation de Films", layout="wide")

st.title("🎬 AAAAAAAA de Recommandation de Films")
st.write("Décrivez le film que vous souhaitez voir, et nous vous proposerons des recommandations !")

# Champ de saisie pour l'utilisateur
user_input = st.text_area("Décrivez votre film idéal (ex: un film d'action avec des super-héros et de l'humour) :", "")

if st.button("Obtenir des recommandations"):
    if user_input:
        recommendations = get_movie_recommendations(user_input, top_n=5)  # Enlever df_movies
        st.subheader("🎥 Voici les 5 films les plus proches de votre description :")
        
        for i, r in enumerate(recommendations, 1):
            st.markdown(f"**{r['title']}**")
            st.write(r['overview'])
            st.write("---")
        
        # Ajout d'un système de feedback
        st.subheader("Avez-vous aimé ces recommandations ?")
        feedback = st.radio("Donnez votre avis :", ["Oui, très utile ! 👍", "Non, pas vraiment 👎"])
        
        if feedback:
            st.success("Merci pour votre retour ! Nous améliorerons notre système.")
    else:
        st.warning("Veuillez entrer une description pour obtenir des recommandations.")

# Footer
st.caption("🔍 Système de recommandation de films - Data Science & Machine Learning 🎥")