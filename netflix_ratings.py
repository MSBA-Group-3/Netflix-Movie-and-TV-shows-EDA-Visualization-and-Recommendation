import pandas as pd
netflix_titles = pd.read_csv("Desktop/archive/netflix_titles.csv")
IMDb_movies = pd.read_csv("Desktop/archive/IMDb movies.csv")
IMDb_ratings = pd.read_csv("Desktop/archive/IMDb ratings.csv")

movie_ratings = pd.merge(IMDb_movies, IMDb_ratings, how = "outer", on = 'imdb_title_id')
ratings_netflix = pd.merge(movie_ratings, netflix_titles, how = "outer", on = 'title')

ratings_c = ratings_netflix.reindex(columns=['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in'])
ratings_cho = ratings_c[['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in']]

ratings_chos = pd.merge(ratings_cho, netflix_titles, how = "right", on = 'title')

ratings_c = ratings_chos.reindex(columns=['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in'])

netflix_ratings = ratings_c[['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in']]
