from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Scraping IMDB reviews for the movie 'Inception' using Selenium

PATH = "/Users/sauravjayan/Downloads/chromedriver"

driver = webdriver.Chrome(PATH)

driver.get("https://www.imdb.com")

search_bar = driver.find_element_by_id("suggestion-search")
search_bar.clear()
search_bar.send_keys("Inception")
search_bar.send_keys(Keys.RETURN)

link = driver.find_elements_by_link_text("Inception")
link[0].click()


reviews_link = driver.find_element_by_xpath('//*[@id="titleUserReviewsTeaser"]/div/a[2]')
reviews_link.click()

while True:
    try:
        loadMoreButton = driver.find_element_by_xpath('//*[@id="load-more-trigger"]')
        time.sleep(1)
        loadMoreButton.click()
        time.sleep(2)
    except:
        break

r = []

reviews = driver.find_elements_by_class_name("title")

for review in reviews:
    r.append(review.text)
 


# The data frame below has all the rerviews for the movie 'Inception'
inception_reviews = pd.DataFrame({'title':['Inception']*len(r), 'reviews':r})