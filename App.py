from distutils.command.config import config
from email import message

from pickle import TRUE
from flask import Flask, jsonify, render_template, flash, redirect, request, url_for

from Streamer import TwitterClient, TweetAnalyzer

from Pre_Data1 import PreData

config = {
    "DEBUG": True
}

app = Flask(__name__)

app.config.from_mapping(config)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route("/home", methods=["GET", "POST"])
def home():
    tweet = []
    username = str(request.form['username'])
    print("User Name : {}".format(username))
    if request.method == 'POST':
        aa = TwitterClient()
        tweet_analyzer = TweetAnalyzer()
        tc = aa.get_twitter_client_api()
        tweet = tc.user_timeline(screen_name=username, count=200)
        with open("roughTweets.txt", "w", encoding="utf-8") as f:
            f.write(str(tweet))
        df = tweet_analyzer.tweets_to_data_frame(tweet)
        with open("tweet_text.txt", "w", encoding="utf-8") as f:
            f.write(str(df))
        return render_template('index.html')


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == 'GET':
        message = {"In GET Method"}
        return jsonify(message)
    elif request.method == 'POST':
        app.logger.info("In request Method")
        pp = PreData()
        app.logger.info("Fetched Predata class")
        pp.Data_pre()
        app.logger.info("Done with preprocessing")
        pp.Calculate_tfidf()
        h1, p1, h2, p2, h3, p3, h4, p4, h5, p5, h6, p6, h7, p7, h8, p8, h9, p9, Bh, h10, p10, h11, p11, h12, p12, h13, p13, h14, p14 = pp.test()
        # print(c)
        return render_template('result1.html', heading1=h1, para1=p1, heading2=h2, para2=p2, heading3=h3, para3=p3, heading4=h4, para4=p4, heading5=h5, para5=p5, heading6=h6, para6=p6, heading7=h7, para7=p7, heading8=h8, para8=p8, heading9=h9, para9=p9, BigHeading=Bh, heading10=h10, para10=p10, heading11=h11, para11=p11, heading12=h12, para12=p12, heading13=h13, para13=p13, heading14=h14, para14=p14)


@app.route("/about")
def about():
    return "<h1>About Page</h1>"

@app.route("/blog")
def blog():
    return render_template('blog.html')
if __name__ == '__main__':
    app.run()
