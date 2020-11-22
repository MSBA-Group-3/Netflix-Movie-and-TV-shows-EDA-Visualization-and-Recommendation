library(tidyverse)
library(dplyr)

ratings <- read_csv('netflix_ratings.csv')
transformed.data <- read_csv('transformed_data.csv')

print(transformed.data)
colnames(transformed.data)

genres_country <- function(data, country_name) {
  genres_info <- data %>%
    filter(country == country_name)
  
  ggplot(data = genres_info, aes(x=genre)) + geom_bar() + 
    labs(x = 'Genres',
         y = 'Number of Movies',
         title = paste('Number of Movies in ',  country_name))
}
genres_country(transformed.data, "Turkey")


genres_total <- function(data) {
  counts <- table(data$genre)
  barplot(counts, xlab="Genres for all movies", col=c("darkblue","red"))
}
genres_total(transformed.data)


genres_total_pie <- function(data) {
  counts <- table(data$genre)
  pie(counts, labels = data$genre, main="Genres for all movies")
}
genres_total_pie(transformed.data)


