import pandas as pd
import json
netflix_titles = pd.read_csv("Desktop/archive/netflix_titles.csv")
IMDb_movies = pd.read_csv("Desktop/archive/IMDb movies.csv")
IMDb_ratings = pd.read_csv("IMDb ratings.csv")
movie_metadata = pd.read_csv("movies_metadata.csv")
"""
movie_ratings = pd.merge(IMDb_movies, IMDb_ratings, how = "outer", on = 'imdb_title_id')
ratings_netflix = pd.merge(movie_ratings, netflix_titles, how = "outer", on = 'title')

ratings_c = ratings_netflix.reindex(columns=['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in'])
ratings_cho = ratings_c[['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in']]

ratings_chos = pd.merge(ratings_cho, netflix_titles, how = "right", on = 'title')

ratings_c = ratings_chos.reindex(columns=['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in'])

netflix_ratings = ratings_c[['title','year','genre','country','avg_vote','votes','reviews_from_users','reviews_from_critics','total_votes','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1','listed_in']]
"""

#rating_clean = IMDb_ratings.drop(columns=[15:],axis=1,inplace=True)
#movie_clean = movie_metadata.drop(columns=['poster_path','homepage','belongs_to_collection','tagline','original_title'])
movie_ratings = pd.merge(IMDb_ratings, movie_metadata, how='inner', left_on='imdb_title_id',right_on='imdb_id' )
#movie_ratings = pd.merge(rating_clean, movie_clean, how='inner', left_on='imdb_title_id',right_on='imdb_id' )

movie_ratings_clean = movie_ratings.reindex(columns=['title','release_date','video','status','spoken_languages','runtime','production_countries','production_companies','popularity','overview','genres','budget','total_votes','weighted_average_vote','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1'])
#spl = movie_ratings_clean.loc[:,'production_countries']['name']
print(movie_ratings_clean.columns.values)

movie_ratings_clean.to_csv(r'netflix_ratings_large.csv',index=False)
#clean dataset
netflix_ratings = pd.read_csv('netflix_ratings_large.csv')
netflix_ratings = netflix_ratings[netflix_ratings['status']=='Released'].sort_values('release_date',ascending=False)
netflix_ratings = netflix_ratings[1:15001]
netflix_ratings.reset_index(inplace=True)



#1 recommendation
netflix_ratings['vote_average'] = (10*netflix_ratings['votes_10']+9*netflix_ratings['votes_9']+8*netflix_ratings['votes_8']+7*netflix_ratings['votes_7']+6*netflix_ratings['votes_6']+5*netflix_ratings['votes_5']+4*netflix_ratings['votes_4']+3*netflix_ratings['votes_3']+2*netflix_ratings['votes_2']+1*netflix_ratings['votes_1'])/netflix_ratings['total_votes']
C=netflix_ratings['vote_average'].mean()
m=netflix_ratings['total_votes'].quantile(0.9)
q_ratings = netflix_ratings.copy().loc[netflix_ratings['total_votes']>=m]

def weighted_rating(x, m=m, C=C):
    v = x['total_votes']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
q_ratings['score'] = q_ratings.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_ratings = q_ratings.sort_values('score',ascending=False)

#Print the top 15 movies
q_ratings[['title', 'total_votes', 'vote_average', 'score']].head(10)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.barh(q_ratings['title'].head(6),q_ratings['score'].head(6),align='center',color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Score')
plt.title('High Rated Movies')

plt.show()

#2 recommendation
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tfidf = TfidfVectorizer(stop_words='english')
netflix_ratings['overview'] = netflix_ratings['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(netflix_ratings['overview'])
tfidf_matrix.shape


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(netflix_ratings.index, index=netflix_ratings['title']).drop_duplicates()
def get_recommendations_similar_content(title, cosine_sim):
  
    idx = indices[title]
    print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    print(sim_scores)
    movie_indices = [i[0] for i in sim_scores]
    return netflix_ratings['title'].iloc[movie_indices]

get_recommendations_similar_content('The Dark Knight Rises',cosine_sim)







