# SC1015 DSAI Mini-Project AY2021-22

Team 1: Chew Zhi Qi, Koh Jia Wei, Gan Hao Yi

# Table of Contents

- [Welcome Message](#welcome-message)
- [Team 1 Members](#team-1-members)
- [Question/Problem Definition](#questionproblem-definition)
- [Dataset Selection & Preparation](#dataset-selection--preparation)
    - [Kaggle](https://www.kaggle.com/datasets/teyang/singapore-hdb-flat-resale-prices-19902020)
- [Exploratory Data Analysis](#exploratory-data-analysis)
    - [EDA.ipynb](https://github.com/BLTech-py/sc1015/blob/main/EDA.ipynb)
    - [Flat Type](#flat-type)
    - [Town](#town)
    - [General Trend](#general-trend)
    - [Interesting Outlier](#interesting-outlier)
- [Machine Learning](#machine-learning)
    - [ARIMA.ipynb](https://github.com/BLTech-py/sc1015/blob/main/ARIMA.ipynb)
    - [ARIMA Introduction](#arima-introduction)
    - [Best Blk for ML](#best-blk-for-ml)
    - [p, d, q value Optimisation](#p-d-q-value-optimisation)
    - [Obtaining Test Data](#obtaining-test-data)
    - [Future Price Prediction](#future-price-predictionfuture-price-prediction)
- [Insights of Data & Conclusion](#insights-of-data--conclusion)
- [Closing Remarks](#closing-remarks)
- [Version History](#version-history)
- [Extras](#extras)

# Welcome Message

Welcome to team 1's DSAI Mini-Project. This Mini-Project gave our team the opportunity to venture beyond the syllabus of
this course to gain insightful skills and knowledge through the analysis of real world trends and data.

We would also like to thank our TALUO TIANZE for constantly encouraging us to experiment with new methods in
approaching our problems and this Mini-Project would not have been possible without her valuable feedback and expertise
in the field of Data Science.

# Team 1 Members

| Name                 |              Area of Focus               |GitHub Acount|
|----------------------|:----------------------------------------------------:|---|
| Chew Zhi Qi          | Data Collection, Machine Learning, GitHub Repository |@AnthonyChew|
| Koh Jia Wei          | Google Slides, Video Presentation, EDA               |@KohJiaWei|
| Gan Hao Yi           | EDA, Data Clean Up, Google Slides                    |@Bghy99|

# Question/Problem Definition

If you are a indie/big game company coming out with a new game what is the prediceted sales for that game. This piqued our team's curiosity on the sale tread of gaming industry, leading us to our question:

> *What is the best genre or type of game to come out wiht the current trend to maximize profit?*

# Dataset Selection & Preparation

After extensive online searching, our team could find existing dataset that is suitable to our use case. So, we decide to collect our own dataset from [SteamSpy](https://steamspy.com/) which is based on [Steam](https://store.steampowered.com/) but with estimated `Owners`.

However, a few issues were faced while we were during data collection.
1. There is no API that reaturns all released game. So we wrote our own code to do a webscraping on the `appid` of games released from 2008-2022. And with the `appid` collected we covert all `JSON` response to `CSV`.
2. Issues with response from `JSON` that `price` and `initialprice` have the value of `.` which is `0.99`. We had do string matching and change it to `0.99`.
3. Encoding issue. There are games with `English`, `Russian` and `Chinese` names. Thus we need to use `ANSI` encoding instead of `UTF-8`.

For our data preparation, there were a few issues that we have to tackle.
1. Removed unrelated data (including test server, play test)
- accounting
- animation & modeling
- game developlemnt
- video production
- photo editing
- web-publisting
- utilities
- autdio production
- software training
2. One hotting columns including `genre`, `language`(top used language).
Genre
- Casual
- Indie
- Free To Play
- RPG
- Strategy
- Sports
- Simulation
- Racing
- Massivley Multiplayer
Language
- English
- Chinese
- French
- German
- Italian
- Spanish
3. Merging similar columns 
- Action  & adventure -- Act_Adv
- violent & gore & sexual content & nudity -- 18+
4. Converting estimated owners to categorical data 
- 0-20000: 0
- 20001-50000: 1
- 50001-100000: 2
- 100001-200000: 3
- 200001-500000: 4
- 500001-1000000: 5
- 1000001-2000000: 6 
- 2000001-5000000: 7 
- 5000001-10000000: 8 
- 10000001-50000000: 9 
- 50000001-20000000: 10
- 20000001-100000000: 11 
- 100000001-200000000: 12
5.Creating a `review` column which is the score of positive review out of 100%.
> *The formula that we came out is review_score = (positive score / positive score * negativescore) * 100*

