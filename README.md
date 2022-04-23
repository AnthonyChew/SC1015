# SC1015 DSAI Mini-Project AY2021-22

Team 1: Chew Zhi Qi, Koh Jia Wei, Gan Hao Yi

# Table of Contents

- [Welcome Message](#welcome-message)
- [Team 1 Members](#team-1-members)
- [Question/Problem Definition](#questionproblem-definition)
- [Dataset Selection & Preparation](#dataset-selection--preparation)
    - [Game Data(CSV)](https://github.com/AnthonyChew/SC1015/blob/main/GameData_backup_with_review.csv)
    - [SteamSpy Data Collection](https://github.com/AnthonyChew/SC1015/blob/main/1_SteamSpyDataCollection.ipynb)
    - [Data Cleaning](https://github.com/AnthonyChew/SC1015/blob/main/2_DataCleanUp.ipynb)
- [Exploratory Data Analysis](#exploratory-data-analysis)
    - [EDA.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/3_EDA.ipynb)
    - [General Trend](#general-trend)
    - [Diving deeper into genre](#diving-deeper-into-genre)
- [Classification Algorithms](#classification-algorithms)
    - [Decision Tree Classifier](https://github.com/AnthonyChew/SC1015/blob/main/4_DecTree_randomForest.ipynb)
    - [Random Forest Classifier](https://github.com/AnthonyChew/SC1015/blob/main/4_DecTree_randomForest.ipynb)
- [Machine Learning](#machine-learning)
    - [MLP.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/5_MLP.ipynb)
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

We would also like to thank our TA LUO TIANZE for constantly encouraging us to experiment with new methods in
approaching our problems. This Mini-Project would not have been possible without his valuable feedback and expertise
in the field of Data Science.

# Team 1 Members

| Name                 |              Area of Focus               |GitHub Acount|
|----------------------|:----------------------------------------------------------:|------------|
| Chew Zhi Qi          | Data Collection, EDA, Machine Learning, GitHub Repository  |@AnthonyChew|
| Koh Jia Wei          | EDA, Machine Learning, Google Slides, Video Presentation   |@KohJiaWei  |
| Gan Hao Yi           | EDA, Data Clean-Up, Google Slides, Classification Analysis |@ghy99      |

# Question/Problem Definition

If you are a Game Developing company coming out with a new game, you would want to estimate the number of people installing the game on Steam Platform. 
The idea of estimating this piqued our team's curiosity about the sales trend of the gaming industry, leading us to our question:

> *What is the best genre or type of game to come out with the current trend to maximize profit?*

# Dataset Selection & Preparation

After extensive online searching, our team found an existing dataset suitable for our use case. We decided to collect our dataset from [SteamSpy](https://steamspy.com/), which draws their data from [Steam](https://store.steampowered.com/) but with an estimated number of `Owners`.

However, we faced a few issues while we were during data collection.
1. The API does not return all the released games on Steam. Therefore, we wrote our own web scraping code to gather all the `appid` of games released from 2008-2022. And with the `appid` collected, we covert all `JSON` responses to `CSV`.
2. Issues with the data from `JSON` that `price` and `initial price` have the value of `.`,  `0.99`. We had to do string matching and change it to `0.99`.
3. Encoding issues. There are games named in different languages. Thus we had to use `ANSI` encoding instead of `UTF-8`.

For our data preparation, there were a few issues that we had to tackle.
1. Removed unrelated data (including test server, playtest) that were not games:
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
Free to Play

| Genre                    | Language (top used language)       |
|:------------------------:|:----------------------------------:|
|       Casual             |      English                       |
|       Indie              |      Chinese                       |
|       RPG                |      French                        |
|       Strategy           |      German                        |
|       Sports             |      Italian                       |
|       Simulation         |      Spanish                       |
|       Racing             |                                    |
|   Massively Multiplayer  |                                    |
|       Free To Play       |                                    |


3. Merging similar columns 

| Columns to merge         | New Column Name                    |
|:------------------------:|:----------------------------------:|
|   Action & Adventure     |   Act_Adv                          |
|   Violent & Gore & Sexual Content & Nudity  |   18+           |


4. Converting estimated owners to categorical data 

| Estimated Owners             | Maps To                        |
|:----------------------------:|:------------------------------:|
|     0-20000                  |   0                            |
|     20001-50000              |   1                            |
|     50001-100000             |   2                            |
|     100001-200000            |   3                            |
|     200001-500000            |   4                            |
|     500001-1000000           |   5                            |
|     1000001-2000000          |   6                            |
|     2000001-5000000          |   7                            |
|     5000001-10000000         |   8                            |
|     10000001-50000000        |   9                            |
|     50000001-20000000        |   10                           |
|     20000001-100000000       |   11                           |
|     100000001-200000000      |   12                           |

# Exploratory Data Analysis

## For our Exploratory Data Analysis we conducted it in jupyter notebook titled [3_EDA.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/3_EDA.ipynb).

As the focus of our project is to predict the number of owners per game, we first plotted the estimated number of users for the games.

## These are our findings:

##Estimated amount of owners:

<ins>Number of Data Points(games) for each estimated range</ins>:

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

Our team tried to look at games that owned by more than 1 million people. This is the results that we got:
```
Free game(Owned by more than 1m people): 669
Priced games(Owned by more than 1m people): 1967
```


## Diving deeper into genre

For games owned by less than 1 million people, we can see that more than 50% of the games released has the genre tag `Indie` or `Action & Adventure`.

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
###### Disclaimer: Percentages add up to more than 100 as most games have overlapping genres

But when we look at games owned by more than 1 million people, the genre is dominated by `Action & Adventure`, with close to 73% of the games under that genre.

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
###### Disclaimer: Percentages add up to more than 100 as most games have overlapping genres

# Classification Algorithms

After of EDA, we now know what are some of the possible attributes that might affect the number of owners per game, such as `genre`, `game price`, `total language supported` and `game reviews`. 
As our target is to predict how popular a game would be so as to calculate their profit margins, we will now attempt to predict the number of owners per game.

## Decision Tree Classifier

Decision Trees are a non-parametric supervised learning method used for classification. We chose to use [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and the code is stored in a jupyter notebook file titled [4_DecTree _ randomForest.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/4_DecTree_randomForest.ipynb).

For our Decision Tree, we will be using Multi-Variate Decision Tree to predict the number of owners per game, with multiple predictors such as `game reviews`, `game genres`, `available languages`, `game price`.

We started off by separating the dataset into 2 dataframes, an `owner` dataframe and a "Predictors" dataframe. Next, we split it into a train and test dataset with 25% of the dataset as the testing dataset. 

We set the tree depth of the decision tree to 4 initially to check if it works, then increased it to depth 10 to make it more accurate. The tree then yielded a result of 81% accuracy through a 12x12 confusion matrix as we have 12 categories for the number of owners. 


| Dataset type: |  Classification Accuracy | 
|---------------|--------------------------|
| Train Dataset |    0.8084955295180918    |
| Test Dataset  |    0.7569478067202479    |


## Random Forest Classifier

Random Forest Classifier is a tree-based machine learning algorithm that uses multiple Decision Trees to make decisions. It takes a random subset of data for each tree and calculate the output. Next, it calculates the average of the outputs. and use that as the final result. 

The max-depth of each tree was set at 10 and the number of trees used was 1000. 
Random Forest yielded an accuracy of 68% which was not very good. 

| Dataset type: |  Classification Accuracy | 
|---------------|--------------------------|
| Train Dataset |    0.6830710703786962    |
| Test Dataset  |    0.6854676026789317    |

We then used the data set with the `owner` category one-hot encoded. The result yielded an accuracy of 73% but it does not beat the decision tree.

| Dataset type: |  Classification Accuracy | 
|---------------|--------------------------|
| Train Dataset |    0.7365426847192682    |
| Test Dataset  |    0.7175711115594109    |


# Machine Learning

The machine learning model that we have chosen is [MLPClassifier](https://www.analyticsvidhya.com/blog/2020/12/mlp-multilayer-perceptron-simple-overview/) and the jupyter notebook is titled [5_MLP.ipynb](https://github.com/AnthonyChew/SC1015/blob/main/5_MLP.ipynb).

## MLPClassifier Introduction

MLPClassifier stands for Multilayer (ML) Perceptron (P) Classifier which is a statistical analysis model that uses neural network to do classification with an input layer,  multiple hidden layers and an output layer.

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

In order for the MLPClassifier model to give a good prediction, we need to use a model that was pre-trained. However, after extensive amount of research online, we could not find any pre-trained model. Thus our team had to train our own model.

##Value Optimisation

After a few rounds of testing with the number of hidden layers, type of solver and activation function, we have one of the best results with:

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

hidden_layer_sizes was set to 6 because for the hidden layers between the starting and ending layer, we need to fit the 4 different types of predictors that we are using, which are the `positive`, `price`, `total_lang`,`onehot genre`.

For solver, we chose `adam` as our stochastic gradient-based optimizer as it worked well with our dataset, yielding the highest accuracy between the 4 available solvers.

We played around with the rest of the values and the best accuracy we got was 74%.

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
- F1-score: Percentage of predictions that were correct
- Support: Number of actual occurrences  

## Confusion Matrix

![image](https://github.com/AnthonyChew/SC1015/blob/main/img/graphs/MLP_confusion_2022-04-23_10-30-00.png)

From the confusion matrix, we can see that the data's predictions are mostly in `Owner_cat 0` due to most of the data residing there. It make sense that with the same genre and languages supported, it will be predicted as `category 0` instead of categories with higher estimated number of owners. 

# Insights of Data & Conclusion

From this Mini-Project we can learn a few things:

1. Due to most of the data falling under the category of `0-20000`, it resulted in our model having high accuracy on `low owner` category(`0`) and close to non on `high owner` category(>= 6).
2. We can also conclude that there is a high percentage for a game to fall into the estimated range of 0 - 50k owners.

#Things to be done better
1. To increase infomation for the games that we have collected. E.g. (Is it a sequal, Esports, Remaster, Good story, Good gameplay, replayability, Online, MOBA). There are more factors that could affect a game with good sales.
2. Change the model to a outlier predicting model like `Automatic Outlier Detection`. Games that have high amount of estimated owners are consider outliers.


# Closing Remarks

This Mini-Project could not have been done alone. I would like to thanks my teammates for the contribution to the project although we found out that we have used the wrong model. It is truly a good learning experience for me and my teammates. The most important lesson  we learnt is to pay more attention to our EDA so that we choose the right model for the problems we want to solve.


# Extra

Here are some extra EDA, decision tree and random forest jupyter notebooks that our team did to further explore our dataset.

<ins>Jupyter notebooks</ins>
- [Desicion Tree](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_DecTree.ipynb)
- [Random Forest](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_randomforest.ipynb)
- [EDA](https://github.com/AnthonyChew/SC1015/blob/main/Archive/Haoyi_code/HY_EDA_comparison.ipynb)

Reference:
- https://steamspy.com/
- https://seaborn.pydata.org/
- https://it-qa.com/when-to-use-mlpclassifier-in-a-neural-net/
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
