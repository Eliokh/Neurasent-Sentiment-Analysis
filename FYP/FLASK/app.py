import requests
import random
from bs4 import BeautifulSoup

import openai

import time
import datetime

from flask import Flask, render_template, url_for, request
#app = Flask(__name__)
app = Flask(__name__, static_folder='static')


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

sentiment_model = "FYP\Model\sentiment_model.pkl"

with open(sentiment_model, 'rb') as file:
    model = pickle.load(file)


data = pd.read_csv('FYP\Dataset\cleandata.csv')


X = data['Review']
y = data['Rating']

# Convert the rating to binary sentiment labels (positive or negative)
y = y.apply(lambda x: 'positive' if x > 3 else 'negative')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# Load your API key from an environment variable or secret management service
openai.api_key = "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"


class Product:
    pass

class Review:
    pass

def gpt(results, query):
    try:
        reviewsCount = len(results)

        if synthesisOn == False:
            return "Synthesis was turned off"

        if reviewsCount == 0:
            return "Unfortunately something went wrong, the Scraper might be blocked."
        # else:
        #     return "Synthesis was turned off"

        #print(results)

        products = ""
       

        for index, item in enumerate(results):
            products += item.imageAlt + " with " + str(item.pos_percentage) + "% positive reviews"
            if index != len(results) - 1:
                products += " and "


        fullPrompt = "Act as an intelligent product and data analyzer named NeuraSent, you have these " + str(reviewsCount) + query + " products along with the percentage of positive reviews for each : - " + products + " - You will give a summary about your analysis for these products, if a product has a reviews percentage less than 60 it will be considered bad, summarize the name of the products, add some smart info too as well as your opinion on which to choose and why."

 


        response = openai.Completion.create(
            model="text-davinci-003",
            prompt = fullPrompt,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        print(fullPrompt)

        answer = response["choices"][0].text

        index = answer.rfind('\n')
        answer = answer[index+1:]

        return answer
    except:
        return "Synthesis failed"

def scrapingAmazon(headers, query):
    # Define the search term
    query = query.replace(' ', '+')
    search_term = query

    
    # Define the URL to scrape
    url = f"https://www.amazon.com/s?k={search_term}"

    # Make a GET request to the URL
    response = requests.get(url, headers=headers)

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the first product links on the search results page
    product_links = soup.find_all("div", attrs={"data-component-type": "s-search-result"})

    #print(soup)
    if soup is not None and '<title>Sorry! Something went wrong!</title>' in str(soup):
        return "scraperBlocked"


    asins = []
    asinCount = 0
    for element in product_links:
        # ASIN
        product_asin = element.get("data-asin")

        # Image
        try:
            product_img_block = element.select_one("img.s-image")
            product_img = product_img_block.get("src")
        except:
            product_img = ""

        # Title
        try:
            product_name_block = element.select_one('span.a-size-medium')
            product_name = product_name_block.text.strip()
        except:
            product_name = "Title Unavailable"
        
        # Price
        try:
            product_price_block = element.select_one("span.a-offscreen")
            product_price = product_price_block.text.strip()
        except:
            product_price = "Unavailable"

        if asinCount == nbProducts:
            break;
        if product_asin:
            asins.append([product_asin, product_name, product_price, product_img])
            asinCount = asinCount + 1


    review_results = []
    all_reviews = []
    # Loop over the first asins
    for asin in asins:
        if asin[0]:
           
            imageSrc = asin[3]
            imageAlt = asin[1]

            price = asin[2]
      
            nbPos = 0
            nbNeg = 0
            pos_percentage = 0
            neg_percentage = 0
            total_product_reviews = 0
            productReviews = []

            for pageNumber in range(1, nbPages + 1):
                reviews_url = "https://www.amazon.com/product-reviews/" + asin[0] + "?pageNumber=" + str(pageNumber)
                reviews_response = requests.get(reviews_url, headers=headers)
                reviews_soup = BeautifulSoup(reviews_response.content, "html.parser")
                
                review_sections = reviews_soup.select("div.a-section.review")
                

                for section in review_sections:
                    review_text = section.select_one("span.a-size-base.review-text").get_text()
                    review_author = section.select_one("span.a-profile-name").get_text()
                    vectorized_review = vectorizer.transform([review_text])
                    # Make predictions
                    prediction = model.predict(vectorized_review)
                    total_product_reviews = total_product_reviews + 1

                    if (prediction[0] == "positive"):
                        nbPos = nbPos + 1
                    else:
                        nbNeg = nbNeg + 1

                    rev = Review()
                    rev.comment = review_text
                    rev.author = review_author
                    rev.sentiment = prediction[0]
                    productReviews.append(rev)

            if (total_product_reviews != 0):
                pos_percentage = (nbPos / total_product_reviews) * 100
                neg_percentage = (nbNeg / total_product_reviews) * 100

            #print("CREATING PRODUICT")
            prod = Product()
            prod.asin = asin[0]
            prod.imageAlt = imageAlt
            prod.imageSrc = imageSrc
            prod.total_product_reviews = total_product_reviews
            prod.pos_percentage = pos_percentage
            prod.neg_percentage = neg_percentage
            prod.price = price
            prod.productReviews = productReviews

            #print(prod)

            # review_results.append([asin, imageAlt, imageSrc, total_product_reviews, pos_percentage, neg_percentage])
            review_results.append(prod)

    return review_results


def scrapingEbay(headers, query):
    # Define the search term
    query = query.replace(' ', '+')
    search_term = query

    # Define the URL to scrape
    url = f"https://www.ebay.com/sch/{search_term}"

    # Make a GET request to the URL
    response = requests.get(url, headers=headers)

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")

    #print(soup)
    # Find the first product links on the search results page
    prods_ul = soup.find('ul', class_='srp-results')

    #product_links = prods_ul.find_all("div", attrs={"data-component-type": "s-search-result"})
    lis = prods_ul.select('div.s-item__info.clearfix')



    products_urls = []
    asinCount = 0
    for li in lis:
        action_link = li.find('a')['href']
        # if asinCount == 4:
        #     break;
        if action_link:
            #prod_url = action_link.get("href")
            products_urls.append(action_link)
            #asinCount = asinCount + 1




    #print(products_urls)
    review_results = []
    all_reviews = []
    # Loop over the first asins
    for url in products_urls:
        if asinCount == 3:
            break;
        if url:
            try:
                # Get Product page
                #product_url = "https://www.amazon.com/dp/" + asin
                product_response = requests.get(url, headers=headers)
                product_soup = BeautifulSoup(product_response.content, "html.parser")

                #image = product_soup.select_one("img#landingImage")
                image = product_soup.select_one("img.img-scale-down")
                #print(image)
                imageSrc = image["src"]
               

                title = product_soup.select_one("h1.x-item-title__mainTitle span")
                imageAlt = title.getText()


                price_element = product_soup.find('span', attrs={'itemprop': 'price'})
                price = price_element.get('content')

                
                nbPos = 0
                nbNeg = 0
                pos_percentage = 0
                neg_percentage = 0
                total_product_reviews = 0

             
                pos_percentage = random.randint(40,80)
                neg_percentage = 100 - pos_percentage

                prod = Product()
                prod.asin = "asin"
                prod.imageAlt = imageAlt
                prod.imageSrc = imageSrc
                prod.total_product_reviews = total_product_reviews
                prod.pos_percentage = pos_percentage
                prod.neg_percentage = neg_percentage
                prod.price = price

                asinCount = asinCount + 1

                review_results.append(prod)
            except:
                #print("EXCEPTION CAUGHT")
                continue


    return review_results


def scraping(query):
    #headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
        'Accept-Language' : 'en-US,en;q=0.5',
        'Accept-Encoding' : 'gzip, deflate, br',
        'DNT' : '1', # Do Not Track Request Header
        'Connection' : 'keep-alive'
    }


    review_results = scrapingAmazon(headers, query)
    #review_results = scrapingEbay(headers, query)
    print("SCRAPING DONE...")
    if review_results == "scraperBlocked":
        return "scraperBlocked", "Scraper was blocked"
    
    gptResponse = gpt(review_results, query)

    return review_results, gptResponse


@app.route('/')
def index():
 
    return render_template('index.html')

@app.route('/search')
def search():
    try:
        query = request.args.get('query')

        global nbProducts
        global nbPages
        global synthesisOn

        nbProducts = request.args.get('nbProducts')
        nbPages = request.args.get('nbPages')
        synthesisOn = request.args.get('synthesis')

        if not nbProducts:
            nbProducts = 3
        if not nbPages:
            nbPages = 1

        if synthesisOn:
            synthesisOn = True
        else:
            synthesisOn = False

        nbProducts = int(nbProducts)
        nbPages = int(nbPages)

        if nbProducts < 1:
            nbProducts = 3
        if nbPages < 1:
            nbPages = 1


        start_time = time.time()
        reviews, gptResponse = scraping(query)
        if reviews == "scraperBlocked":
            return render_template('index.html', error="Too Many Requests in a short amount of time")
        end_time = time.time()
        exec_time = end_time - start_time
        exec_time = "{:.3f}".format(exec_time)
        return render_template('search.html', reviews=reviews, gptResponse=gptResponse, exec_time=exec_time)
    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)



