#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

from sklearn.preprocessing import MinMaxScaler

from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral
from bokeh.palettes import Spectral6, Magma, Inferno
from bokeh.themes import built_in_themes
from bokeh.io import curdoc

from datetime import date, timedelta
from IPython import get_ipython
from PIL import Image
from streamlit import caching

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity


# In[2]:


st.title('Spotify Playlist Recommender Engine')


# In[3]:


"""
# Music Streaming Analysis
Suppose a group of friends aspires to be successful like Ben&Ben
who have constantly topped the charts in Spotify, what could be done
to help them rapidly increase exposure to finally get their big break?
"""


# In[4]:


image = Image.open('eskwelabs.png')
st.sidebar.image(image, caption='', use_column_width=True)


# In[5]:


section = st.sidebar.radio("Sections",
    ("Methodology","Exploratory Data Analysis","Data and Tools","Modelling for Genre Classfication","Evaluation",\
    "Recommender Engine","Strategy","Authors"))


# In[6]:


if section == "Methodology":
    st.subheader('Methodology')
    st.write('A playlist is recommended based on the results from the KNN and SVM analysis to check the accuracy of the model')


# In[7]:


if section =="Exploratory Data Analysis":
    st.subheader('Exploratory Data Analysis')


# In[8]:


if section == "Data and Tools":
    st.subheader('Data and Tools')
    st.markdown("<ul>"                "<li>One year of data is extracted from Spotify from September 2019 to August 20</li>"                "<li>From the data, the characteristics of viral and mainstay songs are identified based on their audio features.</li>"                "<li>Genre Classification is performed as an input for the Recommender Engine</li>"                "<li>The Recommender Engine is used to suggest which genre can the hypothetical band can jam with</li>"
                "</ul>",unsafe_allow_html=True)
    st.markdown("<table>"                "<tr>"                "<th>Key</th>"                "<th>Description</th>"
                "</tr>"\
                "<tr>"\
                "<td>acousticness</td>"\
                "<td>Confidence measure from 0.0 to 1.0 whether the track is acoustic.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>danceability</td>"\
                "<td>Describes how suitable a track is for dancing based on  \
                musical elements including tempo, rhythm stability, beat strength, and overall regularity.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>energy</td>"\
                "<td>Represents a perceptual measure of intensity and activity.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>instrumentalness</td>"\
                "<td>Predicts whether a track contains no vocals.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>liveness</td>"\
                "<td>Detects the presence of an audience in the recording.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>loudness</td>"\
                "<td>Overall loudness of the track in decibels</td>"\
                "</tr>"\
                "<tr>"\
                "<td>speechiness</td>"\
                "<td>Detects the presence of spoken words in a track.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>valence</td>"\
                "<td>Describes the musical positiveness conveyed by a track.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>tempo</td>"\
                "<td>The overall estimated tempo of a track in beats per minute.</td>"\
                "</tr>"\
                "<tr>"\
                "<td>id</td>"\
                "<td>Spotify ID for the track.</td>"\
                "</tr>"\
                "</table>",unsafe_allow_html=True)


# In[9]:


if section == "Modelling for Genre Classfication":
    st.subheader('Modelling for Genre Classification')
    """
    - *Genres Considered:* Rock, Hiphop/Rap, Country, Jazz, Blues, Classical, Electronic, Folk, New Age, Reggae
    - *Features Considered:* danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo	
    - *Train Test Split Ratio:* (0.8:0.2)
    - *K Fold Cross Validation:* [4,5,8,10]
    - *Neighbors Range:* [2:50]
    """


# In[10]:


if section == "Evaluation":
    st.subheader('Evaluation')
    imageEval = Image.open('evaluatemodel.jpg')
    st.image(imageEval,caption='', use_column_width=True)
    """
    ## Notes
    There are 10 Genres considered in evaluating the model. These are:
    - Rock
    - Hiphop/Rap
    - Country
    - Jazz
    - Blues
    - Classical
    - Electronic
    - Folk 
    - New Age
    - Reggae
    
    These features are evaluated based on the Accuracy Score, with the highest score are the chosen model.
    Model 7 is chosen since it has the highest Accuracy Score of 59% with most inclusive genres of 5.
    Model 8, while having higher accuracy, only has 4 genres. 
    Note that removing features leads to an immediate drop in accuracy.
    """


# In[11]:


if section == "Recommender Engine":
    st.subheader("Recommender Engine")

    st.write("Liked Ben&Ben songs? You might also like these songs.")
    #Embed Spotify Playlist 
    components.html('<iframe src="https://open.spotify.com/embed/playlist/7CCSbojW4Okhhj6K2Xwim9" width="520" height="600" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',height=600)
    '''
        Similarity Measures are used to determine the songs that are to be recommended
    '''
    #read data
    chart_tracks_df=pd.read_csv("data/spotify_daily_charts_tracks_predicted_genres.csv")
    #normalize loudness and tempo
    scaler = MinMaxScaler()
    chart_tracks_df['loudness'] = scaler.fit_transform(chart_tracks_df[['loudness']])
    chart_tracks_df['tempo'] =  scaler.fit_transform(chart_tracks_df[['tempo']])
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',                'liveness', 'valence', 'tempo']
    seed_track_data = chart_tracks_df[chart_tracks_df['track_name']=='Maybe The Night'].iloc[0]
    
    # Eucledian
    chart_tracks_df['euclidean_dist'] = chart_tracks_df.apply(lambda x: euclidean_distances(x[feature_cols].values.reshape(-1, 1),                                                                  seed_track_data[feature_cols].values.reshape(-1, 1))                                                                  .flatten()[0], axis=1)
    st.write('Euclidean Distribution')
    #get top 10 nearest to seed_track_data
    recommendation_df = chart_tracks_df[chart_tracks_df['track_id']!=seed_track_data['track_id']].sort_values('euclidean_dist')[:10]
    recommendation_df[['track_name','artist_name','euclidean_dist','predicted_genre']+feature_cols]
    chart_tracks_df['manhattan_dist'] = chart_tracks_df.apply(lambda x: manhattan_distances(x[feature_cols].values.reshape(-1, 1),                                                                  seed_track_data[feature_cols].values.reshape(-1, 1))                                                                  .flatten()[0], axis=1)
    fig, ax = plt.subplots()
    ax.hist(chart_tracks_df['euclidean_dist'], bins=20)
    st.pyplot(fig)
    
    # Manhattan
    #get top 10 nearest to seed_track_data
    st.write('Manhattan Distribution')
    recommendation_df = chart_tracks_df[chart_tracks_df['track_id']!=seed_track_data['track_id']].sort_values('manhattan_dist')[:10]
    recommendation_df[['track_name','artist_name','manhattan_dist','predicted_genre']+feature_cols]
    chart_tracks_df['cosine_dist'] = chart_tracks_df.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),                                                                  seed_track_data[feature_cols].values.reshape(1, -1))                                                                  .flatten()[0], axis=1)
    fig, ax = plt.subplots()
    ax.hist(chart_tracks_df['manhattan_dist'], bins=20)
    st.pyplot(fig)
    
    # Cosine
    #get top 10 nearest to seed_track_data
    st.write('Cosine Distribution')
    recommendation_df = chart_tracks_df[chart_tracks_df['track_id']!=seed_track_data['track_id']].sort_values('cosine_dist')[:10]
    recommendation_df[['track_name','artist_name','cosine_dist','predicted_genre']+feature_cols]
    
    fig, ax = plt.subplots()
    ax.hist(chart_tracks_df['cosine_dist'], bins=20)
    st.pyplot(fig)


# In[12]:


if section == "Strategy":
    st.subheader("Strategy")
    """
     - Bean & Bean must collaborate with the OPM artist found by the recommender engine.
     - To broaden their audience they can choose a 2nd genre to add on their portfolio.
    """


# In[13]:


if section == "Authors":
    st.subheader("Authors")
    st.write("Eskwelabs Cohort V")
    st.markdown("<ul>"    "<li>Kenrick Nocom</li>"    "<li>Kemp Po</li>"
    "<li>Janina Elyse Reyes</li>"\
    "<li>William Raymond Revilla</li>"\
    "<li>Franz Taborlupa</li>"\
    "</ul>", unsafe_allow_html=True)

