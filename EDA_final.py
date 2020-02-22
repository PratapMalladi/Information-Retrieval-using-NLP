import os
os.chdir("D:\\Aravinda\\Data Science\\P2_NLP")
#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize, pos_tag, pos_tag_sents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#Adding new stop words
newStopWords = set(STOPWORDS)
stop.extend(newStopWords)
stop.extend(["one","use","bought","got","put","using","still","turn","kind","really","take","","thank","work","well","better","make","see","going","hold","though","either","two","look","good","look","without","please","let","know","im","look","want","anyone","come","need","thank","use","say"])
#define contractions
contractions_dict = {
     'didn\'t': 'did not',
     'don\'t': 'do not',
 }
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)
#import the csv file
qadata=pd.read_csv("D:\\Aravinda\\Data Science\\P2_NLP\\qa_Electronics.csv")
#drop the unixTime as it is a system gerated time and assuming not required for analysis
qadata.drop('unixTime',axis=1, inplace=True)
#Sub dataframe is created with only 2 columns (question and answer)
qadata_sub=qadata[["question","answer"]]
#Coverting both fields to lowercase
qadata_sub=qadata_sub.applymap(lambda s:s.lower() if type(s) == str else s)
#Removed duplicate questions and answers
qadata_sub_nodup=qadata_sub.drop_duplicates(keep='first', inplace=False)
#Removing, punctuations,digits and converting the text to lowercase
def clean_text_round1(text):
    text=text.lower()
    text=expand_contractions(text)#remove contractions
    text=re.sub(r"http\S+", "", text)#remove urls
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('[''""]','',text)
    text=re.sub('\n','',text)
    text=re.sub(' x ','',text)
    return text
round1= lambda x: clean_text_round1(str(x))
#Implement the round1 cleaning on question and answer columns
qadata_sub_nodup_q=pd.DataFrame(qadata_sub_nodup.question.apply(round1))
qadata_sub_nodup_a=pd.DataFrame(qadata_sub_nodup.answer.apply(round1))
#Remove duplicate questions and answers of individual dataframes
qadata_sub_nodup_q=qadata_sub_nodup_q.drop_duplicates(keep='first', inplace=False)
qadata_sub_nodup_a=qadata_sub_nodup_a.drop_duplicates(keep='first', inplace=False)
#Remove Stopwords
qadata_sub_nodup_q['question_withoutstopwords']=qadata_sub_nodup_q['question'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
qadata_sub_nodup_a['answer_withoutstopwords']=qadata_sub_nodup_a['answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#Lemmatization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(str(text))]
qadata_sub_nodup_q['question_lemma'] = pd.DataFrame(qadata_sub_nodup_q['question_withoutstopwords'], columns=['question_withoutstopwords'])
qadata_sub_nodup_a['answer_lemma'] = pd.DataFrame(qadata_sub_nodup_a['answer_withoutstopwords'], columns=['answer_withoutstopwords'])
qadata_sub_nodup_q['question_lemma'] = qadata_sub_nodup_q.question_lemma.apply(lemmatize_text)
qadata_sub_nodup_a['answer_lemma'] = qadata_sub_nodup_a.answer_lemma.apply(lemmatize_text)
## Implement Stemming
#pst = PorterStemmer()
#qadata_sub_nodup_q['question_stem'] = qadata_sub_nodup_q['question_lemma'].apply(lambda x: ' '.join([pst.stem(y) for y in x]))
#qadata_sub_nodup_a['answer_stem'] = qadata_sub_nodup_a['answer_lemma'].apply(lambda x: ' '.join([pst.stem(y) for y in x]))
#Convert the dataframe to list
question_text=qadata_sub_nodup_q['question_lemma'].tolist()
answer_text=qadata_sub_nodup_a['answer_lemma'].tolist()
#Question and Answer WordCloud
wordcloud_question = WordCloud(width=2800,height=2400).generate(str(question_text))
plt.imshow(wordcloud_question)
plt.title("Question WordCloud")
wordcloud_answer = WordCloud(width=2800,height=2400).generate(str(answer_text))
plt.imshow(wordcloud_answer)
plt.title("Answer WordCloud")

#Get Positive words
with open("D:\\Aravinda\\Data Science\\P2_NLP\\PositiveWords.txt","r") as pos:
  poswords = pos.read().split("\n")
#Get Negative Words
with open("D:\\Aravinda\\Data Science\\P2_NLP\\NegativeWords.txt","r") as neg:
            negwords = neg.read().split("\n")
#Positive Wordcloud
question_text_pos = " ".join ([w for w in question_text if w in poswords])
answer_text_pos = " ".join ([w for w in answer_text if w in poswords])
#question
wordcloud_question_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((question_text_pos))
plt.imshow(wordcloud_question_pos)
plt.title("Positive Question WordCloud")
#answer
wordcloud_answer_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((answer_text_pos))
plt.imshow(wordcloud_answer_pos)
plt.title("Positive Answer WordCloud")
#Negative Wordcloud
question_text_neg = " ".join ([w for w in question_text if w in negwords])
answer_text_neg = " ".join ([w for w in answer_text if w in negwords])
#question
wordcloud_question_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((question_text_neg))
plt.imshow(wordcloud_question_neg)
plt.title("Negative Question WordCloud")
#answer
wordcloud_answer_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((answer_text_neg))
plt.imshow(wordcloud_answer_neg)
plt.title("Negative Answer WordCloud")
qadata_sub_nodup_q.dtypes
qadata_sub_nodup_q['question_lemma']=qadata_sub_nodup_q['question_lemma'].astype(str)
qadata_sub_nodup_a['question_lemma']=qadata_sub_nodup_a['answer_lemma'].astype(str)
corpus_q=qadata_sub_nodup_q['question_lemma']
corpus_a=qadata_sub_nodup_a['answer_lemma']
#Unigram,Bi-Gram and Tri_Gram Analysis
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words_uq = get_top_n_words(corpus_q, n=25)
top_df_uq = pd.DataFrame(top_words_uq)
top_df_uq.columns=["Word", "Freq"]
top_df_uq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\unigram_q.csv", index = None, header=True)
#Convert most freq words to dataframe for plotting bar plot
top_words_ua = get_top_n_words(corpus_a, n=25)
top_df_ua = pd.DataFrame(top_words_ua)
top_df_ua.columns=["Word", "Freq"]
top_df_ua.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\unigram_q.csv", index = None, header=True)
#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df_uq)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title("Most Frequently occuring Question Unigrams")


import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df_ua)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_title("Most Frequently occuring Answer Unigrams")

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Top bigram words
top2_words_bq = get_top_n2_words(corpus_q, n=25)
top2_df_bq = pd.DataFrame(top2_words_bq)
top2_df_bq.columns=["Bi-gram", "Freq"]
top2_df_bq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\bigram_q.csv", index = None, header=True)

top2_words_ba = get_top_n2_words(corpus_a, n=20)
top2_df_ba = pd.DataFrame(top2_words_ba)
top2_df_ba.columns=["Bi-gram", "Freq"]
top2_df_ba.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\bigram_a.csv", index = None, header=True)

#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df_bq)
h.set_xticklabels(h.get_xticklabels(), rotation=45)
h.set_title("Most Frequently occuring Question Bigrams")

import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df_ba)
h.set_xticklabels(h.get_xticklabels(), rotation=45)
h.set_title("Most Frequently occuring Answer Bigrams")

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
#Top Trigram frequencies
top3_words_tq = get_top_n3_words(corpus_q, n=25)
top3_df_tq = pd.DataFrame(top3_words_tq)
top3_df_tq.columns=["Tri-gram", "Freq"]
top3_df_tq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\trigram_tq.csv", index = None, header=True)

top3_words_ta = get_top_n3_words(corpus_a, n=25)
top3_df_ta = pd.DataFrame(top3_words_ta)
top3_df_ta.columns=["Tri-gram", "Freq"]
top3_df_ta.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\trigram_ta.csv", index = None, header=True)

#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df_tq)
j.set_xticklabels(j.get_xticklabels(), rotation=45)
j.set_title("Most Frequently occuring Question Trigrams")

import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df_ta)
j.set_xticklabels(j.get_xticklabels(), rotation=45)
j.set_title("Most Frequently occuring Answer Trigrams")
#Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sentiment_q = qadata_sub_nodup_q['question_withoutstopwords'].apply(lambda x: analyzer.polarity_scores(str(x)))
q_sentiment = pd.concat([qadata_sub_nodup_q['question_withoutstopwords'],sentiment_q.apply(pd.Series)],1)
q_sentiment.describe()

sentiment_a = qadata_sub_nodup_a['answer_withoutstopwords'].apply(lambda x: analyzer.polarity_scores(str(x)))
a_sentiment = pd.concat([qadata_sub_nodup_a['answer_withoutstopwords'],sentiment_a.apply(pd.Series)],1)
a_sentiment.describe()

#TFIDF
#Unigram
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
tvec_uq = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
tvec_weights_uq = tvec_uq.fit_transform(qadata_sub_nodup_q.question_lemma.dropna())
weights_uq = np.asarray(tvec_weights_uq.mean(axis=0)).ravel().tolist()
weights_df_uq = pd.DataFrame({'term': tvec_uq.get_feature_names(), 'weight': weights_uq})
weights_df_uq=weights_df_uq.sort_values(by='weight', ascending=False)
weights_df_uq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\unigram_q_weights.csv", index = None, header=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tvec_ua = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
tvec_weights_ua = tvec_ua.fit_transform(qadata_sub_nodup_a.answer_lemma.dropna())
weights_ua = np.asarray(tvec_weights_ua.mean(axis=0)).ravel().tolist()
weights_df_ua = pd.DataFrame({'term': tvec_ua.get_feature_names(), 'weight': weights_ua})
weights_df_ua=weights_df_ua.sort_values(by='weight', ascending=False)
weights_df_ua.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\unigram_a_weights.csv", index = None, header=True)

#Bi-gram
from sklearn.feature_extraction.text import TfidfVectorizer
tvec_bq = TfidfVectorizer(stop_words='english', ngram_range=(2,2))
tvec_weights_bq = tvec_bq.fit_transform(qadata_sub_nodup_q.question_lemma.dropna())
weights_bq = np.asarray(tvec_weights_bq.mean(axis=0)).ravel().tolist()
weights_df_bq = pd.DataFrame({'term': tvec_bq.get_feature_names(), 'weight': weights_bq})
weights_df_bq=weights_df_bq.sort_values(by='weight', ascending=False)
weights_df_bq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\bigram_q_weights.csv", index = None, header=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tvec_ba = TfidfVectorizer(stop_words='english', ngram_range=(2,2))
tvec_weights_ba = tvec_ba.fit_transform(qadata_sub_nodup_a.answer_lemma.dropna())
weights_ba = np.asarray(tvec_weights_ba.mean(axis=0)).ravel().tolist()
weights_df_ba = pd.DataFrame({'term': tvec_ba.get_feature_names(), 'weight': weights_ba})
weights_df_ba=weights_df_ba.sort_values(by='weight', ascending=False)
weights_df_ba.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\bigram_a_weights.csv", index = None, header=True)

#Tri-gram
from sklearn.feature_extraction.text import TfidfVectorizer
tvec_tq = TfidfVectorizer(stop_words='english', ngram_range=(3,3))
tvec_weights_tq = tvec_tq.fit_transform(qadata_sub_nodup_q.question_lemma.dropna())
weights_tq = np.asarray(tvec_weights_tq.mean(axis=0)).ravel().tolist()
weights_df_tq = pd.DataFrame({'term': tvec_tq.get_feature_names(), 'weight': weights_tq})
weights_df_tq=weights_df_tq.sort_values(by='weight', ascending=False)
weights_df_tq.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\trigram_q_weights.csv", index = None, header=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tvec_ta = TfidfVectorizer(stop_words='english', ngram_range=(3,3))
tvec_weights_ta = tvec_ta.fit_transform(qadata_sub_nodup_a.answer_lemma.dropna())
weights_ta = np.asarray(tvec_weights_ta.mean(axis=0)).ravel().tolist()
weights_df_ta = pd.DataFrame({'term': tvec_ta.get_feature_names(), 'weight': weights_ta})
weights_df_ta=weights_df_ta.sort_values(by='weight', ascending=False)
weights_df_ta.to_csv(r"D:\\Aravinda\\Data Science\\P2_NLP\\\trigram_a_weights.csv", index = None, header=True)

