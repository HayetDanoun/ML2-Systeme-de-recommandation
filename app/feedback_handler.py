# feedback_handler.py
import os
import csv
import datetime

FEEDBACK_FILE = "data/feedback.csv"

def store_feedback(user_query, title, liked, comment="", mode="per_movie"):
    """
    Enregistre le feedback utilisateur dans data/feedback.csv
    """
    fieldnames = ["datetime", "user_query", "title", "liked", "comment", "mode"]
    file_exists = os.path.exists(FEEDBACK_FILE)

    print(f"📂 Tentative d'enregistrement du feedback : {user_query} | {title} | {liked} | {comment}")

    try:
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "datetime": str(datetime.datetime.now()),
                "user_query": user_query,
                "title": title,
                "liked": str(liked),
                "comment": comment,
                "mode": mode
            })

        print(f"✅ Feedback enregistré dans {FEEDBACK_FILE} !")

    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement du feedback : {e}")
