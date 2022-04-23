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
    - [General Trend](#general-trend)
    - [Diving deeper into genre](#diving-deeper-into-genre)
- [Machine Learning](#machine-learning)
    - [MLP.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/MLP.ipynb)
    - [MLPClassifier Introduction](#mlpclassifier-introduction)
    - [Value Optimisation](#value-optimisation)
    - [Classification Report](#classification-report)
    - [Confusion Matrix](#confusion-matrix)
- [Insights of Data & Conclusion](#insights-of-data--conclusion)
- [Closing Remarks](#closing-remarks)
- [Extra](#extra)

# Welcome Message

Welcome to team 1's DSAI Mini-Project. This Mini-Project allowed our team to venture beyond the syllabus of
this course to gain insightful skills and knowledge through analyzing real-world trends and data.

We would also like to thank our TALUO TIANZE for constantly encouraging us to experiment with new methods in
approaching our problems, and this Mini-Project would not have been possible without his valuable feedback and expertise
in the field of Data Science.

# Team 1 Members

| Name                 |              Area of Focus               |GitHub Acount|
|----------------------|:----------------------------------------------------:|---|
| Chew Zhi Qi          | Data Collection, Machine Learning, GitHub Repository |@AnthonyChew|
| Koh Jia Wei          | Google Slides, Video Presentation, EDA               |@KohJiaWei|
| Gan Hao Yi           | EDA, Data Clean-Up, Google Slides                    |@Bghy99|

# Question/Problem Definition

If you are an indie/big game company coming out with a new game, what are the estimated owner for that game and release it on Steam Platform. This piqued our team's curiosity about the sales tread of the gaming industry, leading us to our question:

> *What is the best genre or type of game to come out with the current trend to maximize profit?*

# Dataset Selection & Preparation

After extensive online searching, our team could find an existing dataset suitable for our use case. So, we decided to collect our dataset from [SteamSpy](https://steamspy.com/), which is based on [Steam](https://store.steampowered.com/) but with estimated `Owners`.

However, we faced a few issues while we were during data collection.
1. No API returns all released games. So we wrote our code to do a web scraping on the `appid` of games released from 2008-2022. And with the `appid` collected, we covert all `JSON` responses to `CSV`.
2. Issues with response from `JSON` that `price` and `initial price` have the value of `.`,  `0.99`. We had to do string matching and change it to `0.99`.
3. Encoding issue. There are games with `English`, `Russian` and `Chinese` names. Thus we need to use `ANSI` encoding instead of `UTF-8`.

For our data preparation, there were a few issues that we had to tackle.
1. Removed unrelated data (including test server, playtest)
- accounting
- animation & modeling
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

```
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=6, learning_rate='adaptive',
              learning_rate_init=0.01, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=3, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
```

- hidden_layer_sizes: 6
- activation: relu
- solver: adam

Hidden layer sizes= 6 because `n_layers - 2` we used `positive`, `price`, `total_lang`,`onehot genre`.For solver `adam` was chose because it work well with our large dataset. The rest of the values we just played around to get the best accuracy which is 0.74.

##Classification Report

|       |Precision  |  Recall | F1-score  | Support| 
|-------|---------|-----------|--------|-------|
|0      |  0.84   |   0.98    |  0.90  |   8445|
|1      |  0.32   |   0.64    |  0.43  |    504|
|2      |  0.40   |   0.18    |  0.25  |   1602|
|3      |  0.33   |   0.09    |  0.14  |    817|
|4      |  0.00   |   0.00    |  0.00  |    224|
|5      |  0.30   |   0.35    |  0.32  |    538|
|6      |  0.00   |   0.00    |  0.00  |    142|
|7      |  0.00   |   0.00    |  0.00  |     13|
|8      |  0.00   |   0.00    |  0.00  |     89|
|9      |  0.00   |   0.00    |  0.00  |      2|
|10     |  0.00   |   0.00    |  0.00  |      7|

- Precision: Accuracy of predictions
- Recall: Fraction that were correctly identified
- F1-score: Percent of predictions were correct
- Supoort: Number os actual occurrences  

##Confusion Matrix

![image](https://github.com/AnthonyChew/SC1015/blob/main/img/graphs/MLP_confusion_2022-04-23_10-30-00.png)

From the confusion matrix we can see that the data predict mostly are in `Owner_cat 0` due to most of the data resides there. It make sense that with the same genre, and language support it will be predicted as 0 instead of a higher estimated owner. 

# Insights of Data & Conclusion

From this Mini-Project we can learn a few things:

1. Due to most of the data falls under the category of `0-20000` resutling our model has high accuracy on low owner category(`0`) and close to non on high owner category(>= 6).
2. We can also conclude that there is a high percentage for a game to fall into the estimated range of 0 - 50k owners.

#Things to be done better
1. To increase infomation for the games that we have collected. E.g. (Is it a sequal, Esports, Remaster, Good story, Good gameplay, replayability, Online, MOBA). There are more factors that could affect a game with good sales.
2. Change the model to a outlier predicting model like `Automatic Outlier Detection`. Becuase games that have high amount of estimated owners are consider outliars.

# Closing Remarks

This Mini-Project couldn't be done with one mans' effort but the whole team. I would like to thanks my teammates for the contribution to the project although we found out that we have used the wrong model. It is truely a good learning experice for me and my teamates. And the most important lesson that we have learn is to pay more atthention to our EDA so that we choose the right model for the problems we want to solve.

#Extra

Here are some extra EDA, decision tree and random forest jupyter notebooks that our team did to further explore our dataset.

<ins>Jupyter notebooks</ins>
- [Desicion Tree](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_DecTree.ipynb)
- [Random Forest](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_randomforest.ipynb)
- [EDA](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_EDA_comparison.ipynb)

Refrence:
- https://steamspy.com/
- https://seaborn.pydata.org/
- https://it-qa.com/when-to-use-mlpclassifier-in-a-neural-net/
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

