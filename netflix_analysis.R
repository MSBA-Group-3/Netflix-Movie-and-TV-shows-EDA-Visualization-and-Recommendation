library(tidyverse)
library(dplyr)
library(RColorBrewer)

ratings <- read_csv('netflix_ratings.csv')
transformed.data <- read_csv('transformed_data.csv')

print(transformed.data)
colnames(transformed.data)


# sort the bars
genres_country <- function(data, country_name) {
  genres_info <- data %>%
    filter(country == country_name)
  
  ggplot(data = genres_info, aes(x=genre)) + geom_bar() + 
    labs(x = 'Genres',
         y = 'Number of Movies',
         title = paste('Number of Movies in ',  country_name))
}
genres_country(transformed.data, "United States")

# change labels vertically
genres_total <- function(data) {
  counts <- table(data$genre)
  barplot(counts, main="Genres for all movies", col=c("darkblue"), las=2)
}
genres_total(transformed.data)


genres_total_pie <- function(data) {
  counts <- table(data$genre)
  n <- length(unique(transformed.data$genre))
  qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
  col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

  pie(counts,
      labels = NA, 
      main="Genres for all movies",
      col=sample(col_vector, n))
  legend("left",legend=unique(transformed.data$genre),bty="n",
         fill=col_vector, inset = 0.7, y.intersp = 0.3)
}
genres_total_pie(transformed.data)

