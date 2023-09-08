#!/usr/bin/env python
# coding: utf-8

# # Analyzing User Sentiments and Performance of Threads: An Instagram App Reviews Study

# ![thread.jpeg](attachment:thread.jpeg)

# ## Importing Libraries
# 
# 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('C:/Users/hemak/Downloads/threads_reviews.csv')


# ## Analysis

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.isnull().sum()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.nunique()


# In[14]:


df['source'].unique()


# In[15]:


df['source'].value_counts()


# ## Data Visualisation

# In[16]:


plt.figure(figsize=(5,6))
sns.countplot(df['source'], data = df, palette = 'hls')
plt.show()


# In[17]:


plt.figure(figsize=(5,5))
counts = df['source'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('hls'))
plt.title('Source of Reviews')
plt.show()


# In[18]:


fig = go.Figure(data=[go.Bar(x=df['source'].value_counts().index, y=df['source'].value_counts())])
fig.update_layout(title='Source of Reviews',xaxis_title="Source",yaxis_title="Count")
fig.show()


# In[19]:


counts = df['source'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title='Source of Reviews')
fig.show()


# In[20]:


df['rating'].unique()


# In[21]:


df['rating'].value_counts()


# In[22]:


plt.figure(figsize=(5,6))
sns.countplot(df['rating'], data = df, palette = 'hls')
plt.show()


# In[23]:


plt.figure(figsize=(6, 5))
counts = df['rating'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('hls'))
plt.title('Rating')
plt.show()


# In[24]:


fig = go.Figure(data=[go.Bar(x=df['rating'].value_counts().index, y=df['rating'].value_counts())])
fig.update_layout(title='Rating',xaxis_title="Rating",yaxis_title="Count")
fig.show()


# In[25]:


counts = df['rating'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title='Rating')
fig.show()


# In[26]:


df['review_date'] = pd.to_datetime(df['review_date'])


# In[27]:


df['review_date']


# In[28]:


df1 = df.copy()


# In[29]:


df1.set_index('review_date', inplace=True)
#if you want to do time analysis date should come in index position


# ## If we have date in data. We can do trend analysis

# In[30]:


df1


# In[31]:


start_date = df1.index.min()
end_date = df1.index.max()


# In[32]:


print("Start Date of Reviews:", start_date)
print("End Date of Reviews:", end_date)


# In[33]:


week_delta = pd.Timedelta(days=6)
current_date = start_date


# In[34]:


while current_date <= end_date:
    start_week = current_date
    print(start_week)
    end_week = current_date + week_delta
    print(end_week)
    current_week_data = df1[(df1.index >= start_week) & (df1.index < end_week)]
    print(current_week_data)
    weekly_counts = current_week_data.resample('D').size()
    print("hi",weekly_counts)
    fig = px.bar(weekly_counts, x=weekly_counts.index, y=weekly_counts.values,
                 labels={'x': 'Date', 'y': 'Number of Reviews'},
                 title=f'Reviews for Week {start_week.strftime("%Y-%m-%d")} to {end_week.strftime("%Y-%m-%d")}')
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()
    current_date += week_delta


# In[35]:


daily_counts = df1.resample('D').size()
daily_counts


# In[36]:


fig = go.Figure()
fig.add_trace(go.Bar(x=daily_counts.index, y=daily_counts.values,
                     marker_color='skyblue'))
fig.update_layout(title='Number of Reviews Day-wise',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# fig = go.Figure()
# for source_name, source_data in df1.groupby('source'):
#     fig.add_trace(go.Scatter(x=source_data.resample('D').size().index, y=source_data.resample('D').size().values,
#                             mode='lines', name=source_name))
# 
# fig.show()

# In[37]:


fig = go.Figure()
for source_name, source_data in df1.groupby('source'):
    fig.add_trace(go.Scatter(x=source_data.resample('D').size().index, y=source_data.resample('D').size().values,
                            mode='lines', name=source_name))
fig.update_layout(title='Number of Reviews Day-wise',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# In[38]:


fig = go.Figure()
for source_name, source_data in df1.groupby('source'):
    fig.add_trace(go.Scatter(x=source_data.resample('D').size().index, y=source_data.resample('D').size().values,
                            mode='markers', name=source_name))
fig.update_layout(title='Number of Reviews Day-wise',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# In[39]:


fig = go.Figure()
for rating_val, rating_data in df1.groupby('rating'):
    fig.add_trace(go.Scatter(x=rating_data.resample('D').size().index, y=rating_data.resample('D').size().values,
                             mode='lines', name=f'Rating {rating_val}'))
fig.update_layout(title='Number of Reviews Day-wise by Rating',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# In[40]:


fig = go.Figure()
for rating_val, rating_data in df1.groupby('rating'):
    fig.add_trace(go.Scatter(x=rating_data.resample('D').size().index, y=rating_data.resample('D').size().values,
                             mode='lines+markers', name=f'Rating {rating_val}'))
fig.update_layout(title='Number of Reviews Day-wise by Rating',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# In[41]:


fig = go.Figure()
for rating_val, rating_data in df1.groupby('rating'):
    fig.add_trace(go.Scatter(x=rating_data.resample('D').size().index, y=rating_data.resample('D').size().values,
                             mode='markers', name=f'Rating {rating_val}'))
fig.update_layout(title='Number of Reviews Day-wise by Rating',
                  xaxis_title='Date',
                  yaxis_title='Number of Reviews',
                  xaxis_tickangle=-45)
fig.show()


# In[42]:


df['rating']=df['rating'].map({1:-1,2:-1,3:0,4:1,5:1})


# In[43]:


df


# In[44]:


df_new = df[['review_description', 'rating']]


# In[45]:


df_new 


# In[46]:


df_new['rating'].unique()


# In[47]:


df_new['rating'].value_counts()


# In[48]:


fig = go.Figure(data=[go.Bar(x=df_new['rating'].value_counts().index, y=df_new['rating'].value_counts())])
fig.update_layout(title='Rating',xaxis_title="Rating",yaxis_title="Count")
fig.show()


# In[49]:


counts = df_new['rating'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title='Rating')
fig.show()


# # NLP starts

# ## Data Preprocessing

# In[50]:


def clean_text(text):
    text = text.lower()
    return text.strip()


# In[51]:


df_new


# In[52]:


df_new.review_description = df_new.review_description.apply(lambda x: clean_text(x))


# In[53]:


df_new


# In[54]:


df_new.review_description


# In[55]:


import string
string.punctuation


# In[56]:


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# In[57]:


df_new.review_description = df_new.review_description.apply(lambda x:remove_punctuation(x))


# In[58]:


df_new.review_description = df_new.review_description.apply(lambda x: x.lower())


# In[59]:


df_new.review_description


# In[60]:


import re


# In[61]:


def tokenization(text):
    tokens = re.split(' ',text)
    return tokens


# In[62]:


df_new.review_description = df_new.review_description.apply(lambda x: tokenization(x))


# In[63]:


df_new.review_description


# In[64]:


import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords


# In[65]:


stopwords[0:10]


# In[66]:


def remove_stopwords(text):
    output= " ".join(i for i in text if i not in stopwords)
    return output


# In[67]:


df_new.review_description = df_new.review_description.apply(lambda x:remove_stopwords(x))


# In[68]:


df_new.review_description


# In[69]:


def clean_text(text):
    text = re.sub('\[.*\]','', text).strip() 
    text = re.sub('\S*\d\S*\s*','', text).strip()  
    return text.strip()


# The `clean_text` function you've provided is a Python function for text preprocessing. It takes an input text and performs two regular expression-based operations to clean and preprocess the text. Let's break down each line of the function:
# 
# 1. `text = re.sub('\[.*\]','', text).strip()`
#    - `re.sub(pattern, replacement, string)` is a function from the `re` module in Python for performing regular expression substitution.
#    - `\[\.*\]` is a regular expression pattern that matches square brackets and everything inside them. In regular expressions, square brackets have special meaning, so we need to escape them with a backslash (`\[` and `\]`) to match them literally.
#    - `.*` is a wildcard that matches any character (except for a newline character) zero or more times. So, `\[.*\]` matches everything inside square brackets.
#    - `''` is the replacement string, which is empty. This means that any text inside square brackets will be replaced with nothing, effectively removing it.
#    - `text.strip()` is used to remove any leading or trailing whitespace from the modified text.
# 
# 2. `text = re.sub('\S*\d\S*\s*','', text).strip()`
#    - `\S` is a regular expression pattern that matches any non-whitespace character.
#    - `\d` is a regular expression pattern that matches any digit (0-9).
#    - `\S*\d\S*` matches any sequence of characters that contains at least one digit but no whitespace in between.
#    - `\s*` matches any whitespace characters (including spaces, tabs, or line breaks) zero or more times.
#    - `''` is the replacement string, which is empty. This means that any sequence of characters that contains at least one digit but no whitespace in between will be replaced with nothing.
#    - `text.strip()` is used to remove any leading or trailing whitespace from the modified text.
# 
# Overall, this function is designed to clean and preprocess text by removing two types of patterns:
# 
# 1. Text enclosed within square brackets, including the square brackets themselves.
# 2. Any sequence of characters that contains at least one digit but no whitespace in between.
# 
# After applying both regular expression substitutions and stripping any remaining whitespace, the function returns the cleaned and preprocessed text. This can be useful for tasks like removing noise or unwanted patterns from text data before further analysis or natural language processing tasks.

# In[70]:


df_new.review_description = df_new.review_description.apply(lambda x: clean_text(x))


# In[71]:


df_new.review_description


# In[72]:


import spacy
nlp = spacy.load('en_core_web_sm')


# import spacy: This line imports the spaCy library, which is a popular natural language processing (NLP) library in Python. spaCy provides tools and resources for various NLP tasks, including tokenization, named entity recognition, part-of-speech tagging, and more.
# 
# nlp = spacy.load('en_core_web_sm'): This line loads a specific spaCy language model named 'en_core_web_sm'. spaCy provides pre-trained language models for several languages, and 'en_core_web_sm' is a small English language model that is commonly used for basic NLP tasks. It's a good choice for tasks like tokenization and part-of-speech tagging when you don't need a larger, more computationally intensive model.

# In[73]:


stopwords = nlp.Defaults.stop_words
def lemmatizer(text):
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if not token.text in set(stopwords)]
    return ' '.join(sent)


# In[76]:


df_new.review_description =  df_new.review_description.apply(lambda x: lemmatizer(x)) 


# In[75]:


df_new.review_description


# In[78]:


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)


# re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
# 
# re.sub(pattern, replacement, string, flags) is a function from the re module in Python for performing regular expression substitution.
# 
# r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b' is the regular expression pattern used to match URLs. Let's break it down further:
# 
# (https|http)? matches the optional "https" or "http" at the beginning of a URL.
# :\/\/ matches the colon and double forward slashes that typically appear after "https://" or "http://".
# (\w|\.|\/|\?|\=|\&|\%)* matches any combination of word characters, dots, forward slashes, question marks, equals signs, ampersands, and percentage signs. This part of the pattern allows it to match the rest of the URL.
# \b is a word boundary anchor. It ensures that the regular expression matches only complete URLs and not partial matches within words.
# '' is the replacement string, which is empty. This means that any text matching the URL pattern will be replaced with nothing, effectively removing it.
# 
# vTEXT is the input text from which URLs will be removed.
# 
# flags=re.MULTILINE is an optional flag that allows the regular expression to match URLs across multiple lines in the input text.
# 
# return(vTEXT): The function returns the modified text with URLs removed.

# In[79]:


df_new.review_description = df_new.review_description.apply(lambda x: remove_urls(x))


# In[80]:


df_new.review_description


# In[81]:


def remove_digits(text):
    clean_text = re.sub(r"\b[0-9]+\b\s*", "", text)
    return(text)


# In[82]:


df_new.review_description = df_new.review_description.apply(lambda x: remove_digits(x))


# In[83]:


df_new.review_description


# In[84]:


def remove_digits1(sample_text):
    clean_text = " ".join([w for w in sample_text.split() if not w.isdigit()]) 
    return(clean_text)


# In[85]:


df_new.review_description = df_new.review_description.apply(lambda x: remove_digits1(x))


# In[86]:


df_new.review_description


# In[87]:


def remove_emojis(data):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" 
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', data)


# emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
# 
# re.compile(pattern, flags) is a function from the re module in Python that compiles a regular expression pattern into a regular expression object for efficient pattern matching.
# 
# The pattern "[u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"]+" is a regular expression pattern that matches a range of Unicode code points corresponding to emojis. These code point ranges cover a variety of common emojis used in text.
# 
# flags=re.UNICODE is an optional flag that indicates that the pattern should be applied to Unicode strings. This is necessary because emojis are represented using Unicode characters.
# 
# re.sub(emoji_pattern, '', data)
# 
# re.sub(pattern, replacement, string) is a function from the re module in Python for performing regular expression substitution.
# 
# emoji_pattern is the regular expression pattern defined earlier, which matches emojis.
# 
# '' is the replacement string, which is empty. This means that any text matching the emoji pattern will be replaced with nothing, effectively removing emojis from the input text (data).
# 
# return re.sub(emoji_pattern, '', data): The function returns the modified text with emojis removed.

# In[88]:


df_new.review_description = df_new.review_description.apply(lambda x: remove_emojis(x))


# ### Cleaned Data

# In[89]:


df_new


# # Data Visualisation (Text Data)

# In[90]:


import wordcloud


# In[91]:


from wordcloud import WordCloud
data = df_new.review_description
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[92]:


from wordcloud import WordCloud
data = df_new[df_new['rating'] == 1]['review_description']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[93]:


data = df_new[df_new['rating']==0]['review_description']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[94]:


data = df_new[df_new['rating'] == -1]['review_description']
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data))
plt.imshow(wc)
plt.axis('off')
plt.show()


# # Converting text into numerics

# In[95]:


from sklearn.feature_extraction.text import TfidfVectorizer


# The TfidfVectorizer is a popular text feature extraction method used in natural language processing (NLP) and machine learning. It stands for "Term Frequency-Inverse Document Frequency Vectorizer." This vectorizer is part of the scikit-learn library in Python and is used to convert a collection of raw text documents into a matrix of TF-IDF features. Let's break down what TF-IDF and the TfidfVectorizer do:
# 
# Term Frequency (TF): Term Frequency measures the frequency of a term (word) in a document. It is calculated as the number of times a term appears in a document divided by the total number of terms in that document. The idea is to give higher weight to words that appear frequently within a document because they are likely to be important for that document.
# 
# Inverse Document Frequency (IDF): Inverse Document Frequency measures the importance of a term in the entire corpus (collection of documents). It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term. Terms that are common across many documents receive a lower IDF score, whereas terms that are rare and specific to a few documents receive a higher IDF score.
# 
# TF-IDF: The TF-IDF score combines both Term Frequency and Inverse Document Frequency to assign a weight to each term in a document relative to its importance in the entire corpus. It is calculated as TF * IDF.
# 
# The TfidfVectorizer takes a list of text documents as input and performs the following steps:
# 
# Tokenization: It breaks down each document into individual words or tokens.
# TF-IDF Calculation: It calculates the TF-IDF score for each word in each document.
# Vectorization: It creates a numerical matrix where each row represents a document, and each column represents a unique word (term) from the entire corpus. The values in the matrix are the TF-IDF scores.

# In[96]:


tf1=TfidfVectorizer()
data_vec=tf1.fit_transform(df_new['review_description'])


# In[97]:


data_vec


# In[98]:


import pickle


# In[99]:


with open('tfidf_vectorizer.pkl','wb') as model_file:
    pickle.dump(tf1,model_file)


# In[103]:


y=df_new['rating'].values


# ## Using Classification 

# In[101]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


# In[104]:


X_train,X_test,y_train,y_test=train_test_split(data_vec,y,test_size=0.2,stratify = y, random_state=42)


# ## Treating with Imbalance data

# In[105]:


from imblearn.over_sampling import SMOTE


# SMOTE stands for "Synthetic Minority Over-sampling Technique." It is a technique used in machine learning and particularly in the context of imbalanced datasets to address the problem of class imbalance. Class imbalance occurs when one class (the minority class) has significantly fewer samples than another class (the majority class).
# 
# SMOTE works by generating synthetic examples for the minority class to balance the class distribution. It does this by creating synthetic instances that are similar to the existing minority class instances. Here's how SMOTE works:
# 
# Selecting a Minority Instance: SMOTE starts by randomly selecting a minority instance from the dataset.
# 
# Finding Nearest Neighbors: It then identifies the k-nearest neighbors of the selected instance within the minority class. The value of k is a parameter that you can specify.
# 
# Creating Synthetic Instances: SMOTE generates synthetic instances by interpolating between the selected instance and its k-nearest neighbors. It creates new instances along the line segments connecting the selected instance to its neighbors. The number of synthetic instances generated is also a parameter that you can control.
# 
# Adding Synthetic Instances: Finally, the synthetic instances are added to the dataset, effectively balancing the class distribution.
# 
# The goal of SMOTE is to increase the number of minority class samples in a way that doesn't introduce too much noise into the dataset. By creating synthetic examples that are similar to real instances, SMOTE aims to improve the classifier's ability to learn the minority class and make better predictions on imbalanced datasets.

# In[106]:


smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)


# In[107]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[108]:


sv = SVC()


# In[111]:


model_sv=sv.fit(X_balanced, y_balanced)
y_pred = model_sv.predict(X_test)


# In[113]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
    
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
    
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)
    
print('\n')


# In[114]:


model_filename='svm_model.pkl'
with open(model_filename,'wb') as model_file:
    pickle.dump(model_sv,model_file)


# In[115]:


sv = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
ad = AdaBoostClassifier()

models = [sv, dt, rf, ad]

accuracies = []

for model in models:
    print('Results for the model:', model.__class__.__name__)
    model.fit(X_balanced, y_balanced)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', cm)
    
    report = classification_report(y_test, y_pred)
    print('Classification Report:\n', report)
    
    print('\n')
    
    accuracies.append(accuracy)

print('List of Accuracies:', accuracies)


# In[116]:


model_names = ['SVC', 'DecisionTree', 'RandomForest', 'AdaBoost']
fig = go.Figure(data=go.Bar(x=model_names, y=accuracies))
fig.update_layout(title='Model Accuracies',
                  xaxis_title='Model',
                  yaxis_title='Accuracy',
                  yaxis_tickformat='.2%', 
                  yaxis_range=[0, 1],  
                  xaxis_tickangle=0)
fig.show()


# In[ ]:




