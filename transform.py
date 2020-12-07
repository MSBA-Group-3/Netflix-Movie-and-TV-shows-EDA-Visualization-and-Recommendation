import csv
import pandas as pd

netflix_ratings = pd.read_csv("netflix_ratings.csv")

title = netflix_ratings.loc[:, "title"]
year = netflix_ratings.loc[:, "year"]
genre = netflix_ratings.loc[:, "genre"]
avg_vote = netflix_ratings.loc[:, "avg_vote"]
country = netflix_ratings.loc[:, "country"]


df = pd.concat([title, year, genre, avg_vote, country], axis = 1)


titles = []
years = []
genres = []
countries = []
votes = []
for index, row in df.iterrows():
	for genre in row.genre.split(","):
		for country in row.country.split(","):
			titles.append(row.title)
			years.append(row.year)
			genres.append(genre)
			votes.append(row.avg_vote)
			countries.append(country)


headerRowOne = ['title', 'year', 'genre', 'avg_vote', 'country']
output_filename = 'transformed_data.csv'
output_file = open(output_filename, 'w')

writer = csv.writer(output_file)
writer.writerow(headerRowOne)

rows = zip(titles, years, genres, votes, countries)

for row in rows:
	writer.writerow(row)

print("END")






