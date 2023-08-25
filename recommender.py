import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_and_clean_data(file_path, selected_features):
    try:
        movies_data = pd.read_csv(file_path)
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
        return movies_data
    except FileNotFoundError:
        print("Error: The file could not be found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading and cleaning the data: {e}")
        return None

def calculate_similarity(selected_features, combined_features):
    try:
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(selected_features)
        similarity = cosine_similarity(feature_vectors)
        return similarity
    except Exception as e:
        print(f"An error occurred while calculating similarity: {e}")
        return None

def find_similar_movies(movie_name, movies_data, similarity):
    try:
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not find_close_match:
            return ["No close match found for the entered movie name."]
        
        close_match = find_close_match[0]
        
        if 'index' not in movies_data.columns:
            return ["Error: 'index' column not found in the dataset."]
        
        matching_movies = movies_data[movies_data.title == close_match]
        if matching_movies.empty:
            return [f"No movie found with title: {close_match}"]
        
        # Check if the index column exists and contains valid values
        if 'index' not in matching_movies.columns or len(matching_movies['index']) == 0:
            return ["Error: 'index' values are missing or invalid."]
        
        index_of_the_movie = matching_movies.iloc[0]['index']
        
        # Check if the index value is within bounds
        if index_of_the_movie >= similarity.shape[0]:
            return ["Error: Index is out of bounds for similarity calculation."]
        
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        return sorted_similar_movies
    except Exception as e:
        return [f"An error occurred while finding similar movies: {e}"]


def get_movie_suggestions(sorted_similar_movies, movies_data, limit=30):
    suggestions = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < limit:
            suggestions.append(title_from_index)
    return suggestions

def main():
    try:
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        movie_name = input('Enter your favourite movie name: ')
        
        movies_data = load_and_clean_data('movies.csv', selected_features)
        if movies_data is None:
            return
        
        combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                            movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
        
        similarity = calculate_similarity(selected_features, combined_features)
        if similarity is None:
            return
        
        sorted_similar_movies = find_similar_movies(movie_name, movies_data, similarity)
        if not sorted_similar_movies:
            return
        
        movie_suggestions = get_movie_suggestions(sorted_similar_movies, movies_data)
        
        print('Movies suggested for you:\n')
        for i, movie_title in enumerate(movie_suggestions, start=1):
            print(f'{i}. {movie_title}')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()