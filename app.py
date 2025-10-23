# =======================================
# 🎬 MOVIE RECOMMENDER - STREAMLIT APP
# =======================================

import pickle
import streamlit as st
import requests

# 1️⃣ Function to fetch movie poster from TMDB API
def fetch_poster(movie_id):
    # API call to get movie details and poster path
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# 2️⃣ Function to recommend similar movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movie_names = []
    recommended_movie_posters = []
    
    # Get top 5 similar movies (excluding the selected one)
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
        
    return recommended_movie_names, recommended_movie_posters

# 3️⃣ Load the trained model and similarity data
movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# 4️⃣ Streamlit UI setup
st.header('🎬 Movie Recommender System')

# Dropdown for user to select a movie
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Type or select a movie:", movie_list)

# When button clicked, show 5 recommendations
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    
    # Create 5 columns to display movie names & posters
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(recommended_movie_names[i])
            st.image(recommended_movie_posters[i])
