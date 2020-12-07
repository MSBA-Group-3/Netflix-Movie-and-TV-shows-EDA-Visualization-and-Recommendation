import pandas as pd
import regex as re
import csv

dat = pd.read_csv("netflix_ratings_large.csv")

for index, row in dat.iterrows():
    if type(row['production_countries']) != str:
        dat.drop(index, inplace=True)

def extract_genre(a):
    return re.findall("{'id':\s\d+,\s'name':\s'([A-za-z\s]+)'}",a)

def extract_country(a):
    return re.findall("{'\w+':\s'\w+',\s'name':\s'([A-za-z\s]+)'}",a)


dat['genres'] = dat['genres'].apply(extract_genre)
dat['production_countries'] = dat['production_countries'].apply(extract_country)


title = dat.loc[:, "title"]
year =dat.loc[:, "release_date"]
genre = dat.loc[:, "genres"]
country = dat.loc[:, "production_countries"]
avg_vote = dat.loc[:, "weighted_average_vote"]


df = pd.concat([title, year, genre, avg_vote, country], axis = 1)


titles = []
years = []
genres = []
countries = []
votes = []

for index, row in df.iterrows():
	for genre in row.genres:
		for country in row.production_countries:
			titles.append(row.title)
			years.append(row.release_date)
			genres.append(genre)
			votes.append(row.weighted_average_vote)
			countries.append(country)


headerRowOne = ['title', 'year', 'genre', 'avg_vote', 'country']
output_filename = 'transformed_data_2.csv'
output_file = open(output_filename, 'w')

writer = csv.writer(output_file)
writer.writerow(headerRowOne)

rows = zip(titles, years, genres, votes, countries)

for row in rows:
	writer.writerow(row)

df2 = pd.read_csv('transformed_data_2.csv')












