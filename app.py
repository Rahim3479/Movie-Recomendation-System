import gradio as gr
import recommender as rec
theme = gr.Theme.from_hub("freddyaboulton/dracula_revamped")

input=['text']
def recommend(input):
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    movies_data = rec.load_and_clean_data('movies.csv', selected_features)
    if movies_data is None:
        return
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                        movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    similarity = rec.calculate_similarity(selected_features, combined_features)
    if similarity is None:
        return
    sorted_similar_movies = rec.find_similar_movies(input, movies_data, similarity)
    if not sorted_similar_movies:
        return
    movie_suggestions = rec.get_movie_suggestions(sorted_similar_movies, movies_data)
    return "\n".join(movie_suggestions)
#create Gradio App
demo=gr.Interface(fn=recommend,inputs=input,outputs="text",title="Find Similar Movies",theme=theme,description='Enter a movie name and it will show you similar movies.')
demo.launch(debug=True,share=True) 