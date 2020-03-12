
# Import libraries
import tweepy # Twitter
import numpy
import pandas
import matplotlib
from textblob import TextBlob, Word # Text pre-processing
import re # Regular expressions
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # Word clouds
from PIL import Image
from pyspark.sql import SparkSession # Spark
from twitter_auth import * # Import Twitter authentication file
from nltk.stem import PorterStemmer



# Twitter authentication
def twitter_auth():
    # This function has been completed for you
    # It uses hardcoded Twitter credentials and returns a request handler
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    return tweepy.API(auth)


# Retrieve Tweets
def get_tweets():
    # This function has been completed for you
    # It creates a Tweet list and extracts Tweets
    account = '' # You can change this to any Twitter account you wish
    extractor = twitter_auth() # Twitter handler object
    tweets = []
    for tweet in tweepy.Cursor(extractor.user_timeline, id = account).items():
        tweets.append(tweet)
    print('Number of Tweets extracted: {}.\n'.format(len(tweets)))
    return tweets


# Create dataframe
def make_dataframe(tweets):
    # This function should return a dataframe containing the text in the Tweets
    data = []
    for tweet in tweets:
        data.append(tweet.__dict__['text'])
    return pandas.DataFrame(data = data, columns = ['Tweets'])


# Pre-process Tweets
def clean_tweets(data):
    # This function has been completed for you
    # It pre-processes the text in the Tweets and runs in parallel
    spark = SparkSession    .builder    .appName("PythonPi")    .getOrCreate()
    sc = spark.sparkContext
    paralleled = sc.parallelize(data)
    return paralleled.map(text_preprocess).collect()

# Pre-process text in Tweet
def text_preprocess(tweet):
    # This function should return a Tweet that consists of only lowercase characters,
    # no hyperlinks or symbols, and has been stemmed or lemmatized
    # Hint: use TextBlob and Word(tweet) and look up which functions you can call
    tweet = str(tweet).lower()
    tweet = re.sub(r'http\S+?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    replace = '[^a-zA-Z0-9 \n]'
    tweet = str(tweet).replace(replace,'')
    st = PorterStemmer()
    temp = tweet
    temp_tweet = ''
    for word in temp.split():
        temp_tweet = temp_tweet + st.stem(word) + ' '
    tweet = temp_tweet
    return tweet

# Retrieve sentiment of Tweets
def generate_sentiment(data):
    # This function has been completed for you
    # It returns the sentiment of the Tweets and runs in parallel
    spark = SparkSession.builder.appName("PythonPi").getOrCreate()
    sc = spark.sparkContext
    paralleled = sc.parallelize(data)
    return paralleled.map(data_sentiment).collect()

# Retrieve sentiment of Tweet
def data_sentiment(tweet):
    # This function should return 1, 0, or -1 depending on the value of text.sentiment.polarity
    text = TextBlob(tweet).sentiment.polarity
    if text > 0:
        return 1
    elif text == 0:
        return 0
    else:
        return -1


# Classify Tweets
def classify_tweets(data):
    # Given the cleaned Tweets and their sentiment,
    # this function should return a list of good, neutral, and bad Tweets
    good_tweets = data[data['sentiment'] == 1]
    neutral_tweets = data[data['sentiment'] == 0]
    bad_tweets = data[data['sentiment'] == -1]
    return [good_tweets, neutral_tweets, bad_tweets]

# Create word cloud
def create_word_cloud(classified_tweets) :
    # Given the list of good, neutral, and bad Tweets,
    # create a word cloud for each list
    # Use different colors for each word cloud
    good_tweets = classified_tweets[0]
    neutral_tweets = classified_tweets[1]
    bad_tweets = classified_tweets[2]
    good_words = []
    good_words.extend(good_tweets['cleaned_tweets'])
    good_words = ' '.join(good_words)
    bad_words = []
    bad_words.extend(bad_tweets['cleaned_tweets'])
    bad_words = ' '.join(bad_words)
    neutral_words = []
    neutral_words.extend(neutral_tweets['cleaned_tweets'])
    neutral_words = ' '.join(neutral_words)
    stopwords = set(STOPWORDS)
    good_cloud = WordCloud(width = 800, height = 500).generate(good_words)
    neutral_cloud = WordCloud(width = 800, height = 500).generate(neutral_words)
    bad_cloud = WordCloud(width = 800, height = 500).generate(bad_words)
    produce_plot(good_cloud, "Good.png")
    produce_plot(neutral_cloud, "Neutral.png")
    produce_plot(bad_cloud, "Bad.png")

# Produce plot
def produce_plot(cloud, name):
    # This function has been completed for you
    matplotlib.pyplot.axis("off")
    matplotlib.pyplot.imshow(cloud, interpolation='bilinear')
    fig = matplotlib.pyplot.figure(1)
    fig.savefig(name)
    matplotlib.pyplot.clf()

# Task 01: Retrieve Tweets
tweets = get_tweets()

# Task 02: Create dataframe
df = make_dataframe(tweets)

# Task 03: Pre-process Tweets
df['cleaned_tweets'] = clean_tweets(df['Tweets'])

# Task 04: Retrieve sentiments
df['sentiment'] = generate_sentiment(df['cleaned_tweets'])

# Task 05: Classify Tweets
classified_tweets = classify_tweets(df)

# Task 06: Create Word Cloud
create_word_cloud(classified_tweets)

