import Streamer

from Streamer import TwitterClient,TweetAnalyzer
tweet_analyzer=TweetAnalyzer()
aa = TwitterClient()
tc = aa.get_twitter_client_api()
tweet = tc.user_timeline(screen_name='MaryamNSharif',count=2000)
with open("tweeeeeeets.txt", "w", encoding="utf-8") as f:
    f.write(str(tweet))
df = tweet_analyzer.tweets_to_data_frame(tweet)
with open("tweet_text.txt", "w", encoding="utf-8") as f:
    f.write(str(df))


