# Sentiment-Analysis-of-Reddit-post

## Background

In real world data science project, NLP (natural language process) is an important topic as lots of customer or market information are hided in customer reviews, social media posts or text conversations. 

Analyzing textual data can be a very challenge problem since language is very flexible and hard to be presented in a structed and mathematic way. Fortunately, we’ve got a bunch of useful tools nowadays, such as the well-known NLTK package, as well as the Word Embedding techniques.

In this project, I tried to use the most popular NLP analysis tools and Classification models to perform a “Sentiment analysis”. 

## Installation 

This project has used “WordCloud”.

`!pip install wordcloud`

This project also used “LIWC” lexicon, a proprietary dictionary. Its purchase link is here:[purchase](https://www.receptiviti.com/liwc-api-get-started)

The notebook I shared in this repository is just a function that using the LIWC lexicon to calculate each emotion’s percentage in a giving post. This is not an official LIWC product nor is it in any way affiliated with the LIWC development team or Receptiviti.

## Project Introduction

The text data used in this project is raw textual data extracted through Reddit API. It contains posts from 2018-10-01 to 2019-01-31 in a subreddit, “loseit”. This subreddit is a place for people who want to lose weight to mark their progress or share their experience, so people can encourage each other.

Each reddit post has a score. According to the explanation in [Reddit official Q&A webpage](https://www.reddit.com/wiki/faq#wiki_how_is_a_submission.27s_score_determined.3F), the score is the difference between “upvote” and “downvote” that the post gets. If a post got a negative or zero score, this post is disliked by the majority. While if a post got a positive score, that means people like the post is more people who don’t like it.

I labeled all posts into two classes: “unwelcome” for posts with score that equal or less than 0, “supportive” for posts with score that equal or more than 1. And the final goal of this project is to train a classifier to correctly find out those “unwelcome” post, as those posts are highly likely containing negative or hate speech that will bring negative atmosphere to this subreddit, which is against this subreddit’s principle.

## Data Cleaning

According to reddit instruction, there should be 38 variables for each post in the raw data, however a very large portion of posts didn’t have exactly 38 variables or have variable values in wrong position. By discarding those non-consisted records and removing irrelevant records such as post from “Automoderator” and outlier, there are 137627 records left.

## Feature Extracting

I have used three ways to extract features in this project.

-	LIWC: a popular word count program that developed and maintained by Pennebaker et al. The version I used in this project is 2007, in this version, there are 64 emotion categories. LIWC API will assign each word in a text into one or more emotion categories based on the program rule, and each score of the emotion is the total times that the specific emotion appears in the result divided by the total number of words in the text. The detail introduction is here [How it works](http://liwc.wpengine.com/how-it-works/)
-	NLTK - Vader module: Vader uses a lexicon of words to find which ones are positives or negatives. It also takes into account the context of the sentences to determine the sentiment scores. Vader will delivery 4 scores: negative, neutral, positive and compound. Compound score is calculated based on the other three scores. 
The compound score range in [-1,1], the more near to -1, the text contains more negative information, vice versa.
Unlike Doc2vec, the Vader module contains normalization process within it, and it also take into the upper case words and punctuations to determine the tone of a text, so it can be processed with the original post content. 

-	Doc2vec: it is an word embeding technique to present text documents as a vector.  It is a generalizing of the word2vec method, which take the text context into the consideration. According to its paper, combining both of the “DM” and “DBOM” model results will deliver better result, so I processed in this way in this project. This moduel can be found in [gensim package](https://radimrehurek.com/gensim/models/doc2vec.html )

## Exploratory data analysis

- Distribution and Doxplot of post score. We can see that the score values are obviously positive skewness.

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/box-score.png)
![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/dist-score.png)

- Distribution of posts length. Most of the post have a length range [0, 180]

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/post%20len.png)

- Wordcloud plots for class “unwelcome” and “supportive”. These two plots proves that the most frequent words in these two classes are different.

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/word-sup.png)

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/word-unw.png)

- Distribution of “vader compound score” of class “unwelcome” and “supportive”. From this plot, one can see that the vader score of "supportive" class more located in the postive direction, while the vader score of "unwelcome" class centered in negative direction.

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/vad.png)

- Pie chart of the ratio of class “unwelcome” to “supportive”. This chart clearly indicate that this dataset is imbalanced, and the class "unwelcome" is the minority one.

![image](https://github.com/alice-heqi/Sentiment-Analysis-of-Reddit-post/blob/master/image/ratio.png)

## Imbalanced data approach

From the distribution of two class records above, one can see the “unwelcome” class only accounts for 3% of total records, so it’s severely imbalanced. 
Models that trained on imbalanced data is highly possible to have high model accuracy but can’t predict the minority class. In this project, correctly pinpointing the “unwelcome” class, also the minority class, is the main objective, therefore, this issue needs to be addressed first.

I have tested three ways to deal with the imbalanced data: 1) random up-sampled the “unwelcome” class; 2) SMOTE (Synthetic Minority Oversampling Technique)
 to up sample the “unwelcome” class by creating new and synthetic record with nearest neighbors algorithm; 3) under-sample the “supportive” class records.
After comparing the model score, “under-sample” approach works best in this project.

## Modeling

Among the three ways that I used to extract text features, both LIWC and Vader belongs to lexicon approach, while Doc2Vec is neural network approach, so I run same models to the two different approach to compare the results. The metrics I used in this project are AUC score, F1 score, recall score and precision score. These four scores could give me a relatively full understanding of each model’s performance. Especially the recall score and precision score. If the recall score is high, that means the model can correctly predict most of the records that we care about, while precision score can tell us how many false positive records that the model will also generate.

|Model|AUC-score|F1-score|recall|precision|
|-----|---------|--------|------|---------|
|Random Forest (LIWC,Vader) - original imbalanced sample|0.63|0.0027|	0.0052|	0.1429|
|Random Forest (LIWC,Vader) - upsample|0.64|	0.047|	0.036|	0.067|
|Random Forest (LIWC,Vader) - SMOTE|	0.62|	0.057|	0.052|	0.063|
|Random Forest (LIWC,Vader) - undersample|	0.66|	0.096|	0.609|	0.052|
|Logistic Regression (LIWC,Vader) - undersample|	0.66|	0.097|	0.579|	0.053|
|SVC (LIWC,Vader) - undersample|	0.66|	0.095|	0.626|	0.051|
|Random Forest (Doc2vec) - undersample|	0.73|	0.12|	0.64|	0.066|
|Logistic Regression (Doc2vec) - undersample|	0.74|	0.12|	0.678|	0.066|
|SVC (Doc2vec) - undersample|	0.75|	0.123|	0.695|	0.068|
|SVC (Doc2vec,Vader) - undersample|	0.75|	0.123|	0.7|	0.067|
|SVC (Doc2vec,LIWC) - undersample|	0.75|	0.123|	0.694|	0.068
|SVC (Doc2vec,Vader,LIWC) - undersample|	0.75|	0.122|	0.7|	0.067|

## Conclusion

Model with LIWC and Vader didn’t deliver an acceptable AUC score, while Doc2vec preforms much better.

The approach that using Doc2vec features, downsized sample and SVC model delivers the highest AUC score, but it pays a cost that the number of “False Positive” also increased. Whether it’s worth to pay this cost depends on the real working environment. 

Besides, combining Doc2vec features with LIWC and Vader didn’t improve the accuracy score. 

Generally speaking, if the features that generated from the textual data can accurately reflect the meaning of the text, the model accuracy will be high. In this project, even the highest AUC score is still under 0.8. To improve the model accuracy, the focus should be exploring other NLP feature engineering techniques to improve the feature quality.












