# # ================================
# # 🎬 MOVIE RECOMMENDER - MODEL PART
# # ================================

# # Import necessary libraries
# import pandas as pd
# import ast
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# # 1️⃣ Load both datasets (movies + credits)
# movies = pd.read_csv('TMDB_movie_dataset_v11.csv')
# credits = pd.read_csv('tmdb_5000_credits.csv')

# # 2️⃣ Merge both datasets on 'title' to combine features
# movies = movies.merge(credits, on='title')

# # 3️⃣ Keep only useful columns
# movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# # 4️⃣ Clean data and remove null values
# movies.dropna(inplace=True)

# # 5️⃣ Convert stringified lists (like JSON) into actual Python lists
# def convert(obj):
#     L = []
#     for i in ast.literal_eval(obj):
#         L.append(i['name'])
#     return L

# movies['genres'] = movies['genres'].apply(convert)
# movies['keywords'] = movies['keywords'].apply(convert)
# movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
# movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

# # 6️⃣ Create a 'tags' column that combines overview + genres + keywords + cast + crew
# movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# # 7️⃣ Build a new dataframe with only id, title, and tags
# new_df = movies[['movie_id', 'title', 'tags']]

# # 8️⃣ Convert list to single string (lowercase for uniformity)
# new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# # 9️⃣ Convert text into numerical form using TF-IDF
# tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
# vectors = tfidf.fit_transform(new_df['tags']).toarray()

# # 🔟 Calculate similarity between all movies using cosine similarity
# similarity = cosine_similarity(vectors)

# # 1️⃣1️⃣ Save the data for use in the web app
# pickle.dump(new_df, open('model/movie_list.pkl', 'wb'))
# pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

# print("✅ Model training complete and files saved in /model folder!")


# ================================
# 🎬 MOVIE RECOMMENDER - MODEL PART
# ================================

# Import necessary libraries
import pandas as pd
import ast
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1️⃣ Load both datasets (movies + credits)
movies = pd.read_csv('TMDB_movie_dataset_v11.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2️⃣ Merge both datasets on 'title'
movies = movies.merge(credits, on='title')

# 3️⃣ Keep only useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 4️⃣ Drop rows with null values
movies.dropna(inplace=True)

# =========================================
# 🧩 Helper functions for data cleaning
# =========================================
def safe_literal_eval(obj):
    """Safely convert stringified list/dict to Python object."""
    if pd.isna(obj):
        return []
    try:
        return ast.literal_eval(obj)
    except:
        try:
            return json.loads(obj)
        except:
            return []

def extract_names(obj):
    """Extract the 'name' field from list of dicts."""
    names = []
    for i in safe_literal_eval(obj):
        if isinstance(i, dict) and 'name' in i:
            names.append(i['name'])
    return names

def extract_cast(obj):
    """Extract top 3 cast member names."""
    names = []
    for i in safe_literal_eval(obj)[:3]:
        if isinstance(i, dict) and 'name' in i:
            names.append(i['name'])
    return names

def extract_director(obj):
    """Extract the director's name from crew."""
    names = []
    for i in safe_literal_eval(obj):
        if isinstance(i, dict) and i.get('job') == 'Director':
            names.append(i['name'])
    return names

# =========================================
# 🧹 Apply the cleaning functions
# =========================================
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)
movies['cast'] = movies['cast'].apply(extract_cast)
movies['crew'] = movies['crew'].apply(extract_director)

# =========================================
# 🧠 Combine all text features into one column
# =========================================
movies['tags'] = (
    movies['overview'].apply(lambda x: x.split()) +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

# Create new DataFrame
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to single string and lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# =========================================
# 🔢 Convert text into numerical features
# =========================================
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# =========================================
# 💾 Save the model files
# =========================================
os.makedirs("model", exist_ok=True)
pickle.dump(new_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("✅ Model training complete! Files saved in /model folder.")
