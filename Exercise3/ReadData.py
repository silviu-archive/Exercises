import pandas as pd
import missingno as msno
import numpy as np
import dateutil
import datetime

def main():

    #Read csv and input into pandas dataframe
    df = pd.read_csv('tweets.csv', error_bad_lines=False, encoding='ISO-8859-1')

    #Quick summary of the dataframe
    description = df.describe()
    columns = df.columns
    print(columns)
    print(description)
    #One row per tweet
    #Information about: post date, ID, body, retweet(bool), source, inreplytostatusID, inreplytouserID,
    #InreplytoScreenname, number of retweets, number of favourites, hashtags in the tweet, ID of 'place', place name,
    #Place full name, country, place bounding box, place attributes, place contained within, userID, user name,
    #User screen name, user location, user description, user link, user expanded link, user follower count,
    #User friends count, user listed count, user signup date, user tweet count, macro iteration number, place again?

    #Analyze missing values:
    #msno.matrix(df)
    #msno.bar(df)
    #Significant amount of missing values in:
    #TweetInReplyToStatusID, TweetInReplyToUserID, TweetInReplyToScreenName - transform into boolean 'Is the tweet a reply'
    #TweetPlaceID, TweetPlaceName, TweetPlaceFullName, TweetCountry, TweetPlaceBoundingBox, tweet.place
        #Limited information about where a tweet was posted from - transform into boolean potential
    #TweetPlaceAttributes, TweetPlaceContainedWithin - fully missing, delete
    #Some missing values:
    #User Location - fill NaN with 'None', then encode
    #User Description - ???
    #UserLink, UserExpandedLink - Transform into boolean?







    #Create modeling dataframe
    dfModel = pd.DataFrame()
    dfModel['TweetID'] = df['TweetID']

    # Define target as retweet count
    dfModel['RetweetCount'] = df['TweetRetweetCount']

    # Transform tweet retweet flag from boolean to binary(int)
    dfModel['TweetRetweetFlag'] = df['TweetRetweetFlag'].astype(int)

    #Define function to check if a cell is null (0) or not(1)
    def isCellNull(x):
        if x is np.nan:
            return 0
        return 1

    #Transform user inputs (location, description, link) into exists (1) / not exists (0)
    dfModel['UserLocation'] = df['UserLocation'].apply(lambda x: isCellNull(x))
    dfModel['UserDescription'] = df['UserDescription'].apply(lambda x: isCellNull(x))
    dfModel['UserLink'] = df['UserLink'].apply(lambda x: isCellNull(x))

    #Create flag for whether tweet is a reply or not
    dfModel['TweetIsAReply'] = df['TweetInReplyToStatusID'].apply(lambda x: isCellNull(x))


    #Process user signup dates
    #Define function to convert UTC into datetime
    def convertToDatetime(x):
        converted = dateutil.parser.parse(x)
        return converted
    #Convert user singup date and tweet posted time into datetime format
    dfModel['UserSignupDate'] = df['UserSignupDate'].apply(lambda x: convertToDatetime(x))
    dfModel['TweetPostedTime'] = df['TweetPostedTime'].apply(lambda x: convertToDatetime(x))

    #Select the last tweet posting time as a point of reference
    latestTweetTime = dfModel['TweetPostedTime'].max()

    #Select the hour the tweet was posted at (TO BE ENCODED)
    dfModel['TweetTimeslot'] = dfModel['TweetPostedTime'].dt.hour

    #Calculate how many days since a user signed up (from reference tweet)
    dfModel['DaysSinceSignup'] = (latestTweetTime - dfModel['UserSignupDate']).dt.days

    #Calculate how many seconds since a particular tweet (from reference tweet)
    dfModel['SecondsSinceTweet'] = (latestTweetTime - dfModel['TweetPostedTime']).dt.seconds

    #Count number of favorites on a tweet - scaled to time interval since tweet
    dfModel['FavoritesScaledSinceTweet'] = df['TweetFavoritesCount'] / dfModel['SecondsSinceTweet']

    #Count number of user tweets - scaled to time interval since user signed up
    dfModel['UserTweetCountScaledSinceCreation'] = df['UserTweetCount'] / dfModel['DaysSinceSignup']







    #Tweet body - TDIDF matrix




    #Tweet source - remove HTML, create labels, one hot encode

    #Tweet reply to user ID - transform into label that somehow counts top users it replies to?







    #Tweet hashtag - separate hashtags, one hot encode


    #Tweet place id - transform NAN into unknown, one hot encode


    #User ID - too many to encode, potentially not do anything with them?





    #Research macroiterationnnumber




    print('x')





main()