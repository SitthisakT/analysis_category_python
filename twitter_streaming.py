# Import the necessary package to process data in JSON format
import json
# Import the tweepy library
import tweepy

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '1609708140-TFRBugDMYEnrAemqIhAzMK2L81aPW3cQy98azNN'
ACCESS_SECRET = 'lPFXqjEOKONSwMNjyLZjwWgPJUIdffKgmGsV1mEK0uafz'
CONSUMER_KEY = 'IJAQFs9xdIXR3JxgPR32nDvTb'
CONSUMER_SECRET = 'TBQzL6PPyBCQccHlw3Px21M2ccEf7IwiU2blZMSMGO1pj14gHD'

# Setup tweepy to authenticate with Twitter credentials:
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Create the api to connect to twitter with your creadentials
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

# Declare topics for data retrieval
TRACKING_KEYWORDS = [['entertainment'],['technology'],['sport'],['food'],['travel']]
# Declare output file name
OUTPUT_FILE = ['Entertainment.txt','Technology.txt','Sport.txt','Food.txt','Travel.txt']
# Declare number of tweets for retrieval
TWEETS_TO_CAPTURE = 2000
# Declare languages of tweets for retrieval
LANGUAGES = ['en']

# Customize tweepy.SteamListener
class MyStreamListener(tweepy.StreamListener):
    """
    Twitter listener, collects streaming tweets and output to a file
    """
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open(OUTPUT_FILE[i], "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        
        # Stops streaming when it reaches the limit
        if self.num_tweets <= TWEETS_TO_CAPTURE:
            if self.num_tweets % 100 == 0: # Just to see some progress...
                print('Numer of {} tweets captured so far: {}'.format(TRACKING_KEYWORDS[i], self.num_tweets))
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)

for i in range(len(TRACKING_KEYWORDS)):
    # Initialize Stream listener
    l = MyStreamListener(i)

    # Create you Stream object with authentication
    stream = tweepy.Stream(auth, l)

    # Filter Twitter Streams to capture data by the keywords:
    stream.filter(track=TRACKING_KEYWORDS[i], languages=LANGUAGES)