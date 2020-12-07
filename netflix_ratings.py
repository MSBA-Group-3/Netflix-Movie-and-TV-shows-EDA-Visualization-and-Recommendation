import pandas as pd
import json
#netflix_titles = pd.read_csv("Desktop/archive/netflix_titles.csv")
#IMDb_movies = pd.read_csv("Desktop/archive/IMDb movies.csv")
#IMDb_ratings = pd.read_csv("IMDb ratings.csv")
#movie_metadata = pd.read_csv("movies_metadata.csv")
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

movie_ratings_clean = movie_ratings.reindex(columns=['title','id','release_date','video','status','spoken_languages','runtime','production_countries','production_companies','popularity','overview','genres','budget','total_votes','weighted_average_vote','median_vote','votes_10','votes_9','votes_8','votes_7','votes_6','votes_5','votes_4','votes_3','votes_2','votes_1'])
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

#3 recommendation
from surprise import Reader, Dataset, SVD
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import cross_validate
ratings = pd.read_csv('ratings_small.csv')
ratings.drop_duplicates(['movieId','userId'],inplace=True)
ratings['rating'] = ratings['rating'].astype(int)


reader=Reader()
data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
#data.split(n_folds=5)
svd=SVD()
cross_validate(svd,data,measures=['RMSE','MAE'])

trainset = data.build_full_trainset()
svd.fit(trainset)

ratings[ratings['userId']==1]
svd.predict(1,302,3)
svd.predict(1,862,3)


movie_list = ratings['movieId'].unique()
movie_predict = pd.DataFrame({'movieId':movie_list})
movie_metadata = movie_metadata.dropna(subset=['vote_count'])

columns = ['id','title']
movie_all = pd.DataFrame(movie_metadata,columns=columns)

movie_predict_1 = pd.merge(movie_predict,movie_all,how='inner',left_on='movieId',right_on='id')

movie_predict_1['Estimate_Score'] = movie_predict_1['movieId'].apply(lambda x: svd.predict(1,x).est)

movie_predict_1 = movie_predict_1.sort_values('Estimate_Score',ascending=False)

movie_predict_1 = movie_predict_1.drop('id',axis = 1)

movie_predict_1 = movie_predict_1.drop('movieId',axis = 1)

movie_predict_1


#4 recommendation

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']






    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    movies = netflix_ratings.iloc[movie_indices][['title', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, x).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)





#傻逼都会做的
netflix_ratings_small = pd.read_csv('netflix_ratings.csv')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
text = list(set(netflix_ratings_small.genre))
plt.rcParams['figure.figsize'] = (13, 13)
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(str(text))

plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


import datetime
netflix_ratings_partial = netflix_ratings[1:10000]
import plotly.graph_objects as go
topdirs=pd.value_counts(netflix_ratings[:10000]['runtime'])
fig = go.Figure([go.Bar(x=topdirs.index, y=topdirs.values , text=topdirs.values,marker_color='indianred')])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()

sns.set(style="darkgrid")
sns.kdeplot(data=netflix_ratings['runtime'], shade=True)

#countries with highest rate content
joint_data=joint_data.sort_values(by='Rating', ascending=False)
country_count=joint_data['country'].value_counts().sort_values(ascending=False)
country_count=pd.DataFrame(country_count)
topcountries=country_count[0:11]
topcountries
