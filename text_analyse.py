import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import spacy
nlp=spacy.load("en_core_web_lg")
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import itertools


class topic :
    def first_process(df):
        df['processed'] = df['text'].str.lower()
        df['processed']= df['processed'].str.replace('[^\w\s]','', regex=True)
        df['processed'] = df['processed'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        df['processed'] = df['processed'].str.replace('\n', '')
        df['processed'] = df['processed'].str.replace('\t', ' ')
        df['processed'] = df['processed'].str.replace(' {2,}', ' ', regex=True)
        df['processed'] = df['processed'].str.strip()

    def clean_text(text):
    
        cleaned_text = "".join([x for x in text if x not in string.punctuation]) # remove punctuation
        cleaned_text = cleaned_text.lower() # lowercase all characters
        cleaned_text = cleaned_text.strip() # strip whitespace
        cleaned_text = re.sub('[0-9]+', '', cleaned_text)
    
    # remove all emojis
        def deEmojify(inputString):
            return inputString.encode('ascii', 'ignore').decode('ascii')
        cleaned_text = deEmojify(cleaned_text)
    
        tokens = cleaned_text.split(" ") # split string into list of words
    
        stop_words = stopwords.words('english') + stopwords.words('french')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come','l','si','faire','ca'])
        cleaned_text = [token for token in tokens if token not in stop_words] # filter out stopwords
    

        return ' '.join(cleaned_text)

    def lemmatize_sentence(tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        for word, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence

    def delete_most_common_words(df):
        def flatten(t):
            return [item for sublist in t for item in sublist]
        split_it = []

        for i in df['comment']:
            split_it.append(i.split())
        split_it = flatten(split_it)

        Counters_found = Counter(split_it)

        most_occur = Counters_found.most_common(1)
        com = []
        for i in df['comment']:
            com.append(i.replace(most_occur[0][0],''))
        df['comment'] = com

    def convert_df_column_as_freq_dist(dataframe,column,no_common_words=10):

        word_dict = []

        for row in dataframe[column]:
            split_row = str(row).split()
            for word in split_row:
                word_dict.append(word)
            
        title_freqdist = FreqDist(word_dict)
    
        return title_freqdist.most_common(no_common_words)
     
    def convert_df_column_as_freq_dist_bigrams(dataframe,column, reverse_order=True, no_of_bigrams=10):

        bigrams_dict = {}

        for row in dataframe[column]:
            nltk_token = nltk.word_tokenize(row)
            bigram = list(nltk.bigrams(nltk_token))
            for pair_bigram in bigram:
                if pair_bigram in bigrams_dict:
                    bigrams_dict[pair_bigram] += 1
                else:
                    bigrams_dict[pair_bigram] = 1 
    
        sorted_bigrams_dict = {k: v for k, v in sorted(bigrams_dict.items(), 
                                        key=lambda item: item[1],
                                        reverse=reverse_order)}
        return dict(itertools.islice(sorted_bigrams_dict.items(), no_of_bigrams))


def processing(df):
    topic.first_process(df)
    clean_comment = []
    for i in df['processed']:
        clean_comment.append(topic.clean_text(i))
    lema_comment = []
    for i in clean_comment:
        lema_comment.append(' '.join(topic.lemmatize_sentence(i.split())))
    df_clean = pd.DataFrame()
    df_clean['comment'] = lema_comment
    topic.delete_most_common_words(df_clean)
    most_common_words_reviews = dict(topic.convert_df_column_as_freq_dist(df_clean,'comment',no_common_words=20))
    most_common_bigrams = topic.convert_df_column_as_freq_dist_bigrams(df_clean,'comment',
                                                             reverse_order=True,
                                                             no_of_bigrams=20)
    return most_common_words_reviews,df_clean['comment'],most_common_bigrams

