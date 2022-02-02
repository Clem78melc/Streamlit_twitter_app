import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
import gensim.corpora as corpora
import text_analyse
from wordcloud import WordCloud
from text_analyse import processing
import topic_modeling
from topic_modeling import lda_mod,process_words
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import nltk
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from nltk.corpus import stopwords
from matplotlib.patches import Rectangle
stop_words = stopwords.words('english') + stopwords.words('french')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come','l','si','faire','ca'])

st.title("My Twitter Streamlit App")
# For newline
st.write('\n')
st.markdown(
        """
Hello there ! Welcome to your awesome Twitter app.
Just upload your twitter csv here 
""")

imagess_ = Image.open('/home/matrice-ecole/Images/BOEEbgmj_400x400.jpg')
show = st.image(imagess_, use_column_width=True)



st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['csv'] )
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep='\t',on_bad_lines='skip')

option = st.sidebar.selectbox('select twitt topic',('None','Ukraine','Environment', 'Chess','ArrestTrumpNow'))

if option == 'None':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep='\t',on_bad_lines='skip')

if option=='Chess':
  csv_file = open("Chess.csv")
  df = pd.read_csv(csv_file, sep='\t',on_bad_lines='skip') 


if option=='Ukraine':
  csv_file = open("Ukraine.csv")
  df = pd.read_csv(csv_file, sep='\t',on_bad_lines='skip') 

if option=='ArrestTrumpNow':
  csv_file = open("Trump.csv")
  df = pd.read_csv(csv_file, sep='\t',on_bad_lines='skip') 

if option=='Environment':
  csv_file = open("Environment.csv")
  df = pd.read_csv(csv_file, sep='\t',on_bad_lines='skip') 

st.sidebar.title('Analyse')
    
if st.sidebar.button("Click Here to Analyse"):
    
    if uploaded_file is None and option == 'None':
        
        st.sidebar.write("Please upload or choose a Twitter file to Process")
    
    else:
    
        with st.spinner('Analysing ...'):
            st.header("Analysing results: ")
            keys =processing(df)[0].keys()
            values = processing(df)[0].values()
            fig1 = plt.figure(figsize = (10,10))
            plt.title('Most Common Words in Reviews',fontsize=30)
            plt.xlabel('Word',fontsize=10)
            plt.xticks(fontsize=10, rotation=45)
            plt.ylabel('Count',fontsize=10)
            plt.yticks(fontsize=10)
            plt.bar(keys, values)
            st.pyplot(fig1)

            wordcloud_reviews = WordCloud(background_color='white')
            wordcloud_reviews = wordcloud_reviews.generate(' '.join(processing(df)[1].tolist()))
            fig2 = plt.figure( figsize=(10,10) )
            plt.imshow(wordcloud_reviews)
            plt.title("WordCloud - Most Common Words", fontsize=25)
            plt.axis('off')
            st.pyplot(fig2)

            keys_bigrams = [str(key) for key in processing(df)[2].keys()]
            values_bigrams = processing(df)[2].values()

            fig3 = plt.figure(figsize=(20,10))
            plt.title('Most Common Bigrams in Reviews',fontsize=30)
            plt.xlabel('Count',fontsize=25)
            plt.xticks(fontsize=20, rotation=90)
            plt.ylabel('Word',fontsize=25,)
            plt.yticks(fontsize=20)
            plt.tick_params(axis='x',labelsize=20,rotation=45)
            plt.barh(keys_bigrams, values_bigrams)
            st.pyplot(fig3)

            st.success('Done!')


st.sidebar.title('Topic modeling')
    
if st.sidebar.button("Click Here to Process"):

    if uploaded_file is None and option == 'None':
        
        st.sidebar.write("Please upload or choose a Twitter file to Process")
    
    else:
    
        with st.spinner('Processing ...'):
            st.header("Topic Modeling Results: ")
            texts = process_words(df, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            id2word = corpora.Dictionary(texts)
            corpus = [id2word.doc2bow(text) for text in texts]
            ldamodel=lda_mod(corpus,id2word)
            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
            df_topic_sents_keywords = topic_modeling.format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=texts)

            # Format
            df_dominant_topic = df_topic_sents_keywords.reset_index()
            df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
            df_dominant_topic.head(10)

            cloud = WordCloud(stopwords=stop_words,
                            background_color='white',
                            width=2500,
                            height=1800,
                            max_words=10,
                            colormap='tab10',
                            color_func=lambda *args, **kwargs: cols[i],
                            prefer_horizontal=1.0)

            topics = ldamodel.show_topics(formatted=False)

            fig_cloud, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig_cloud.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.pyplot(fig_cloud)


            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

            fig_topic, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):    
                df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
                doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
                ax.hist(doc_lens, bins = 100, color=cols[i])
                ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
                sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
                ax.set(xlim=(0, 25), xlabel='Document Word Count')
                ax.set_ylabel('Number of Documents', color=cols[i])
                ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

            fig_topic.tight_layout()
            fig_topic.subplots_adjust(top=0.90)
            #plt.xticks(np.linspace(0,500,9))
            fig_topic.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
            st.pyplot(fig_topic)

            topics = ldamodel.show_topics(formatted=False)
            data_flat = [w for w_list in texts for w in w_list]
            counter = Counter(data_flat)

            out = []
            for i, topic in topics:
                for word, weight in topic:
                    out.append([word, i , weight, counter[word]])

            df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        


            fig_bar_topic, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
            for i, ax in enumerate(axes.flatten()):
                ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
                ax_twin = ax.twinx()
                ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
                ax.set_ylabel('Word Count', color=cols[i])
                ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
                ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
                ax.tick_params(axis='y', left=False)
                ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
                ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

            fig_bar_topic.tight_layout(w_pad=2)    
            fig_bar_topic.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
            st.pyplot(fig_bar_topic)


            corp = corpus[0:13]
            mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

            fig_topic2, axes = plt.subplots(13-0, 1, figsize=(20, (13-0)*0.95), dpi=160)       
            axes[0].axis('off')
            for i, ax in enumerate(axes):
                if i > 0:
                    corp_cur = corp[i-1] 
                    topic_percs, wordid_topics, wordid_phivalues = ldamodel[corp_cur]
                    word_dominanttopic = [(ldamodel.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
                    ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                                                            fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

           
                    topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
                    ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

                    word_pos = 0.06
                    for j, (word, topics) in enumerate(word_dominanttopic):
                        if j < 14:
                            ax.text(word_pos, 0.5, word,
                                horizontalalignment='left',
                                verticalalignment='center',
                                fontsize=16, color=mycolors[topics],
                                transform=ax.transAxes, fontweight=700)
                            word_pos += .009 * len(word)  # to move the word for the next iter
                            ax.axis('off')
                    ax.text(word_pos, 0.5, '. . .',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color='black',
                        transform=ax.transAxes)       

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.suptitle('Sentence Topic Coloring for Documents: ' + str(0) + ' to ' + str(13-2), fontsize=22, y=0.95, fontweight=700)
            plt.tight_layout()
            st.pyplot(fig_topic2) 
            

            

