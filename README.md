# SC1015 DSAI Mini-Project AY2021-22

Team 1: Chew Zhi Qi, Koh Jia Wei, Gan Hao Yi

# Table of Contents

- [Welcome Message](#welcome-message)
- [Team 1 Members](#team-1-members)
- [Question/Problem Definition](#questionproblem-definition)
- [Dataset Selection & Preparation](#dataset-selection--preparation)
    - [Game Data(CSV)](https://github.com/AnthonyChew/SC1015/blob/main/GameData_backup_with_review.csv)
    - [SteamSpy Data Collection](https://github.com/AnthonyChew/SC1015/blob/main/SteamSpyDataCollection.ipynb)
    - [Data Cleaning](https://github.com/AnthonyChew/SC1015/blob/main/DataCleanUp.ipynb)
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
approaching our problems and this Mini-Project would not have been possible without his valuable feedback and expertise
in the field of Data Science.

# Team 1 Members

| Name                 |              Area of Focus               |GitHub Acount|
|----------------------|:----------------------------------------------------:|---|
| Chew Zhi Qi          | Data Collection, Machine Learning, GitHub Repository |@AnthonyChew|
| Koh Jia Wei          | Google Slides, Video Presentation, EDA               |@KohJiaWei|
| Gan Hao Yi           | EDA, Data Clean Up, Google Slides                    |@Bghy99|

# Question/Problem Definition

If you are a indie/big game company coming out with a new game what is the prediceted sales for that game and realease it on Steam Platform. This piqued our team's curiosity on the sale tread of gaming industry, leading us to our question:

> *What is the best genre or type of game to come out wiht the current trend to maximize profit?*

# Dataset Selection & Preparation

After extensive online searching, our team could find existing dataset that is suitable to our use case. So, we decide to collect our own dataset from [SteamSpy](https://steamspy.com/) which is based on [Steam](https://store.steampowered.com/) but with estimated `Owners`.

However, a few issues were faced while we were during data collection.
1. There is no API that returns all released game. So we wrote our own code to do a web scraping on the `appid` of games released from 2008-2022. And with the `appid` collected we covert all `JSON` response to `CSV`.
2. Issues with response from `JSON` that `price` and `initialprice` have the value of `.` which is `0.99`. We had do string matching and change it to `0.99`.
3. Encoding issue. There are games with `English`, `Russian` and `Chinese` names. Thus we need to use `ANSI` encoding instead of `UTF-8`.

For our data preparation, there were a few issues that we have to tackle.
1. Removed unrelated data (including test server, play test)
- accounting
- animation & modelling
- game development
- video production
- photo editing
- web-publishing
- utilities
- audio production
- software training

2. One hot columns including `genre` and `language`.

| Genre                  | Language (top used language)       |
|:------------------------:|:----------------------------------:|
|       Casual             |   English                          |
|       Indie              |   Chinese                          |
|       RPG                |   French                           |
|       Strategy           |   German                           |
|       Sports             |   Italian                          |
|       Simulation         |   Spanish                          |
|       Racing             |                                    |
|   Massively Multiplayer  |                                    |

3. Merging similar columns 

| Columns                  | New Column Name                    |
|:------------------------:|:----------------------------------:|
|   Action & Adventure     |   Act_Adv                          |
|   Violent & Gore & Sexual Content & Nudity  |   18+           |

5. Converting estimated owners to categorical data 

| Estimated Owners         | Maps To                            |
|:------------------------:|:----------------------------------:|
|       0-20000                |   0                            |
|       20001-50000            |   1                            |
|       50001-100000           |   2                            |
|       100001-200000          |   3                            |
|       200001-500000          |   4                            |
|       500001-1000000         |   5                            |
|       1000001-2000000        |   6                            |
|       2000001-5000000        |   7                            |
|       5000001-10000000       |   8                            |
|       10000001-50000000      |   9                            |
|       50000001-20000000      |   10                           |
|       20000001-100000000     |   11                           |
|       100000001-200000000    |   12                           |

# Exploratory Data Analysis

For our Exploratory Data Analysis we conducted it in a jupyter notebook
titled [EDA.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/EDA.ipynb) and here are our findings:

Estimated amount of owners:

The very first thing that we realized `owners` is an estimated range of user bought the game, therefore our team decided that would be our foucs for our Mini-project.

<ins>Number of Data Points for each estimated range</ins>

```
0-20000                   28246                           
20001-50000               5315                           
50001-100000              2641                         
100001-200000             1860                           
200001-500000             1727                          
500001-1000000            743                          
1000001-2000000           409                          
2000001-5000000           267                          
5000001-10000000          66                           
10000001-50000000         23                            
50000001-20000000         8                          
20000001-100000000        2                      
100000001-200000000       1
```
## General Trend

As we can see from the table above, most games released in steam has owner in the range of 0-20000. 

Our team tried to look at games that owned by more than 1 million people. This is the results we've got:
```
Free game(Owned by more than 1m people): 669
Priced games(Owned by more than 1m people): 1967
```
## Diving deeper into genre

We can see that more than 50% for the games realease had the genre under `Indie` or `Action & Adventure`.

![image](https://github.com/AnthonyChew/SC1015/blob/main/img/graphs/Genre-2022-04-22.png)

<ins>Genre percentage in original data</ins>

```
Indie:                  73.82%
Act_Adv:                63.83%
Casual:                 40.56%
Simulation:             19.51%
Strategy:               19.32%
RPG:                    17.08%
Free to Play:           7.75%
Sports:                 5.07%
Racing:                 3.67%
Massively Multiplayer:  2.83%
18+:                    0.24%
```

But when we look at games owned by more than 1 million people. The trend is dominated by `Action & Adventure` close to 73% of games had the genre.

![image](https://github.com/AnthonyChew/SC1015/blob/main/img/graphs/GenreOverAll-2022-04-22.png)

<ins>Genre percentage in data owned by more than 1 million people</ins>

```
Act_Adv:                73.2%
Indie:                  37.63%
RPG:                    25.13%
Free to Play:           22.29%
Strategy:               21.01%
Simulation:             19.2%
Massively Multiplayer:  12.5%
Casual:                 12.37%
Sports:                 4.64%
Racing:                 2.71%
18+:                    0.0%
```

# Machine Learning

Now that we know some possible attributes that might affect the `owner_cat` of a game, let us attempt to predict the estimated owner of a game given their `genre`, `price`and `total language support`. To let a company decide wheather withe the current coming with such game would bring them profit. The machine learning model that we have chosen is [MLPClassifier](https://www.analyticsvidhya.com/blog/2020/12/mlp-multilayer-perceptron-simple-overview/) and the jupyter notebook is titled [MLP.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/MLP.ipynb).

## MLPClassifier Introduction

MLPClassifier stands for Multilayer (ML) Perceptron (P) Classifier which is a statistical analysis model that uses neural network to do classification with an input layer, hidden layer and an output layer.

MLPClassifier consists of 3 variables which are used to fit the machine learning model:

- hidden_layer_sizes: the number of neurons in the `ith` hidden layer.
- activation: Activation function for the hidden layer
    -   `identity`: no-op activation, useful to implement linear bottleneck, returns f(x) = x
    -   `logistic`: the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
    -   `tanh`: the hyperbolic tan function, returns f(x) = tanh(x).
    -   `relu`: the rectified linear unit function, returns f(x) = max(0, x)
- solver: The solver for weight optimization.
    -   `lbfgs`: is an optimizer in the family of quasi-Newton methods.
    -   `sgd`: refers to stochastic gradient descent.
    -   `adam`: refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

In order for the MLPClassifier model to give a good prediction, we need to have a pre-trained model that predicts but after extensive amound of search online we couldn't find any pre-trained model. Thus our tema had to train our own model.

##Value Optimisation

After a few rounds of playing witht he amount of hidden layer, solver and activation function. We have one of the best results with:

- hidden_layer_sizes: 6
- activation: relu
- solver: adam

For solver `adam` was chose because it work well with our large dataset. The rest of the values we just played around to get the best accuracy up to 0.738
