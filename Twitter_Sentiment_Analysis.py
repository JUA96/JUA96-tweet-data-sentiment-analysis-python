import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
tweets_df = pd.read_csv('data/twitter.csv')
# Take a look at the data
tweets_df.head(15)
tweets_df.info

tweets_df.describe()

# Clean and manipulate data
tweets_df['tweet']
# Drop unwanted columns
tweets_df = tweets_df.drop(['id'], axis=1)

# Explore dataset
# Are there any null elements?
import seaborn as sns
 sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
 
 tweets_df.hist(bins = 30, figsize = (13,5), color = 'r')

 # plot similar figure using seaborn counplot
 sns.countplot(tweets_df['label'], label = 'Count')

 #Check the length of the messages
 tweets_df['length'] = tweets_df['tweet'].apply(len)
tweets_df

tweets_df['length'].plot(bins=100, kind='hist') 

tweets_df.describe()

# Let's see the shortest message 
tweets_df[tweets_df['length'] == 11]['tweet'].iloc[0]

positive = tweets_df[tweets_df]['label']==0]
positive

negative = tweets_df[tweets_df['label']==1]
negative

# word cloud
sentences = tweets_df['tweet'].tolist()
sentences

len(sentences)

sentences_string = " ".join(sentences)

!pip install WordCloud
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_string))

# negative word cloud
negative_list = negative['tweet'].tolist()
negative_sentences_as_string = " ".join(negative_list)
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_string))


# data cleaning 
## removing punctuation and stop words
import string
string.punctuation

Test = 'Good morning beautiful people :)... I am having attempting to write some python code!!'
Test_punc_removed - [ char for char in Test if char not in string.punctuation]
Test_punc_removed

# join the characters to form the string
Test_punc_removed_join = ' '.join(Test_punc_removed)
Test_punc_removed_join

Test_punc_removed = []
for char in Test:
    if char not in string.punctuation:
        Test_punc_removed.append(char)

Test_punc_removed_join = ' '.join(Test_punc_removed)   
Test_punc_removed_join  

# removing stop words continued...
import nltk # Natural Language tool kit 

nltk.download('stopwords')
# download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')

Test_punc_removed_join_clean = for word in Test_punc_removed_join() if word.lower() not in stopwords.words('english')
Test_punc_removed_join_clean # Only important (no so common) words are left

# count vectorization
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

print(vectorizer.get_feature_names())
print(X.toarray())

# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# Let's test the newly added function
tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
print(tweets_df_clean[5]) # show the cleaned up version
print(tweets_df['tweet'][5]) # show the original version

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])

print(vectorizer.get_feature_names())
print(tweets_countvectorizer.toarray())  

tweets_countvectorizer.shape
tweets = pd.DataFrame(tweets_countvectorizer.toarray())

X = tweets
X

y = tweets_df['label']

# Machine Learning:
## Naive-Bayes Classification

X.shape
y.shape

# performing train/test split on the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit the data to the classifier 
from sklearn.naive_bayes import MultinomialNB
# train the model on 31,000 tweets
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# Evaluate model performance
from sklearn.metrics import classification_report, confusion_matrix

# Predict the test set results
y_predict_test - NB_classifier.predict(X_test)
cm = confusion_matrix(y_yest, y_predict_test)
sns.heatmap(cm, annot = True)
# check results and precision of model
print(classification_report(y_test, y_predict_test))

