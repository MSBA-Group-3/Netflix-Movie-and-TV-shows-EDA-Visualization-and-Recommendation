import csv
import pandas as pd

netflix_ratings = pd.read_csv("netflix_ratings.csv")

title = netflix_ratings.loc[:, "title"]
year = netflix_ratings.loc[:, "year"]
genre = netflix_ratings.loc[:, "genre"]
country = netflix_ratings.loc[:, "country"]


df = pd.concat([title, year, genre, country], axis = 1)


titles = []
years = []
genres = []
countries = []
for index, row in df.iterrows():
	for genre in row.genre.split(","):
		for country in row.country.split(","):
			titles.append(row.title)
			years.append(row.year)
			genres.append(genre)
			countries.append(country)


headerRowOne = ['title', 'year', 'genre', 'country']
output_filename = 'transformed_data.csv'
output_file = open(output_filename, 'w')

writer = csv.writer(output_file)
writer.writerow(headerRowOne)

rows = zip(titles, years, genres, countries)

for row in rows:
	writer.writerow(row)

print("END")






