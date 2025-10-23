# Movi_Recommendation

📖 Overview

This project is a content-based movie recommendation system built using Python, Pandas, Scikit-learn, and Streamlit.
It recommends movies similar to the one you like, based on textual metadata such as genres, keywords, cast, and crew.

The model uses TF-IDF vectorization and cosine similarity to find movies that are most similar in content.

🚀 Features

🔍 Recommends top 5 similar movies for any given title

🎭 Combines multiple features like genres, keywords, cast, and crew

🧠 Uses TF-IDF Vectorization for text representation

🧮 Computes cosine similarity for recommendations

🌐 Interactive web interface using Streamlit

🖼️ Fetches movie posters dynamically from TMDB API

🧩 Tech Stack
Layer	Tools / Libraries
Frontend	Streamlit
Backend / Model	Python, Scikit-learn, Pandas, NumPy
Data Source	TMDB 5000 Movie Dataset
API	The Movie Database (TMDB) API
📁 Project Structure
movie_recommendation/
│
├── app.py                      # Streamlit web app
├── notebook_model.py           # Model training and preprocessing script
├── TMDB_movie_dataset_v11.csv  # Movies dataset
├── tmdb_5000_credits.csv       # Credits dataset
├── model/
│   ├── movie_list.pkl          # Serialized movie data
│   └── similarity.pkl          # Cosine similarity matrix
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation

⚙️ Installation & Setup

1️⃣ Clone the repository:

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system


2️⃣ Create a virtual environment:

python -m venv venv


Activate it:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

3️⃣ Install dependencies:

pip install -r requirements.txt


(If you don’t have a requirements.txt, install manually:)

pip install streamlit pandas scikit-learn requests


4️⃣ Run the model training script:

python notebook_model.py


5️⃣ Run the Streamlit web app:

streamlit run app.py


6️⃣ Open your browser at:

http://localhost:8501

🔑 API Key Setup

You need a TMDB API key to fetch movie posters.

Go to https://www.themoviedb.org/settings/api

Create an account and get your API key (v3 auth)

Add it inside app.py:

API_KEY = "your_tmdb_api_key_here"

🧠 How It Works

Data is cleaned and combined from movies and credits datasets.

Text features (overview, genres, keywords, cast, crew) are merged into a single “tags” column.

These tags are converted to numerical vectors using TF-IDF Vectorizer.

Cosine similarity is calculated between all movie vectors.

When a user selects a movie, the top 5 most similar movies are shown with their posters.

📸 Screenshots
Home Page	Recommendations

	

(Replace these links with your actual screenshots!)

💡 Future Improvements

Add collaborative filtering using user ratings

Deploy the app on Streamlit Cloud or Render

Add search bar and filter options (genre/year)

Include movie trailers using YouTube API

🧑‍💻 Author

Jayant Agrawal
🎓 B.Tech CSE | Software Developer | Data Enthusiast
📬 LinkedIn
 • GitHub
