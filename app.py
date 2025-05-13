import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Set Streamlit page config
st.set_page_config(page_title="AI Movie Recommender", layout="centered")

# Sample Movie Data
movies = pd.DataFrame({
    'title': ['Inception', 'Titanic', 'The Matrix', 'The Godfather', 'Avengers'],
    'genres': [['Action', 'Sci-Fi'], ['Romance', 'Drama'], ['Sci-Fi', 'Action'], ['Crime', 'Drama'], ['Action', 'Fantasy']],
    'rating': [8.8, 7.8, 8.7, 9.2, 8.4]
})

# --- Content-Based TF-IDF Recommender (title to title) ---
# Convert genres to text for TF-IDF
movies['genres_str'] = movies['genres'].apply(lambda x: ' '.join(x))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_recommendations(title, n=5):
    if title not in indices:
        return pd.DataFrame(columns=['title', 'genres'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    recs = movies.iloc[movie_indices][['title', 'genres']]
    recs['genres'] = recs['genres'].apply(lambda g: ', '.join(g))
    return recs

# --- Genre-Based Recommender ---
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])
genre_choices = sorted(set(genre for sublist in movies['genres'] for genre in sublist))

def recommend_movie(user_genres):
    if not user_genres:
        return pd.DataFrame()

    user_vector = mlb.transform([user_genres])
    scores = cosine_similarity(user_vector, genre_matrix)[0]
    indices = np.where(scores > 0)[0]
    sorted_indices = indices[np.argsort(scores[indices])[::-1]]
    recommended = movies.iloc[sorted_indices][['title', 'genres', 'rating']].copy()
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    return recommended

# --- Streamlit UI ---
st.title("🎬 AI-Powered Movie Recommender")
st.markdown("Get movie suggestions either by selecting a movie or choosing genres.")

# Tabs: One for each method
tab1, tab2 = st.tabs(["🎞️ Recommend by Movie", "🎯 Recommend by Genre"])

# Tab 1 - Content-Based
with tab1:
    selected_movie = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))
    top_n = st.slider("Number of similar movies to show:", 1, 10, 5)
    if selected_movie:
        results = get_content_recommendations(selected_movie, top_n)
        st.subheader(f"🎥 Movies similar to *{selected_movie}*:")
        st.table(results)

# Tab 2 - Genre-Based
with tab2:
    selected_genres = st.multiselect("Choose your favorite genres:", genre_choices)
    if selected_genres:
        result_df = recommend_movie(selected_genres)
        st.subheader("🎥 Movies matching your genres:")
        st.table(result_df)
    else:
        st.info("👈 Please select at least one genre to see recommendations.")
