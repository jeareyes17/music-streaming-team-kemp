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
    ("Rationale","Methodology","Exploratory Data Analysis","Data and Tools","Modelling for Genre Classfication","Evaluation",\
    "Recommender Engine","Strategy","Authors"))


# In[6]:


if section == "Rationale":
    """
    Bean&Bean is a rising indie Filipino band that are looking for their big break.
    Our team Kemp Records is tasked to help their music make it to the top
    We want to devise a strategy for the band to take for their future music projects in line with those that are consistent in the top music charts and incorporate those qualities in their music.
    Our baseline would be their style which is generally Folk and have a similar music style with Ben&Ben
    """


# In[7]:


if section == "Methodology":
    st.subheader('Methodology')
    imageEval = Image.open('methodology.jpg')
    st.image(imageEval,caption='', use_column_width=True)


# In[8]:


if section =="Exploratory Data Analysis":
    st.subheader('Exploratory Data Analysis')
    df = pd.read_csv('data/merged_charts_tracks_datav2.csv')
    fig, ax = plt.subplots()
    ax = df.corr()
    sns.heatmap(ax)
    st.pyplot(fig)
    df['date'] = pd.to_datetime(df['date'])
    st.markdown('<h3>Characteristics of Viral Songs</h3>',unsafe_allow_html=True)
    df_viral = df[df['is_viral']==1]
    #groupby to remove duplicate entries of the same track_id
    df_viral.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']].mean()
    #assign to new df df_viraltracks for EDA
    df_viraltracks = df_viral.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']].mean()
    #Viral DataFrame Audio Features Distribution
    for col in ['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']:
        fig, ax = plt.subplots()
        ax = sns.distplot(df_viraltracks[col], kde=True)
        plt.title(col)
        plt.ylabel('Frequency')
        plt.show()
        st.pyplot(fig)
    """
    # Audio Features Observations for Viral Tracks:

    popularity = Skewed to the right, obviously most are going to have high popularity being a viral track and in the Top 200. Most are in the 70-90 range. However, what is notable is there are a number of tracks with low popularity which are still considered viral.

    danceability = Skewed to the right, mostly 0.5 - 0.9. Viral tracks tend to be more danceable than the usual charting tracks.

    energy = Almost normally distributed. Both low and high energy tracks can be viral.

    key = Pitch (0 = do, 2 = re, 4 = mi, 5 = fa, 7 = sol, 9 = la, 11 = ti). Well distributed.

    loudness = Skewed to the right, closely matches the distribution for all Spotify tracks. Majority seem to be in the -7.5dB range or higher.

    mode = (Major = 1; Minor = 0) More tracks are using Major modality.

    speechiness = (>0.66 = purely spoken words; 0.33 - 0.66 = both music and speech such as rap; <0.33 music and non-speech tracks)
    Most values are between 0 - 0.1. 

    acousticness = (0 not acoustic; 1 acoustic) Skewed to the left, though while most viral hits are not acoustic, the remaining share of more acoustic songs (>0.12 - 0.9) is more evenly distributed frequency wise.

    instrumentalness = (0 has vocals; 0.5 - 1 no vocals) Majority of the tracks have vocals.

    liveness = (0 not live ; 0.8 - 1 high chance to be live) Majority of the tracks aren't live and probably taken from the album versions.

    valence = (0 sad/angry ; 1 happy/cheerful) Almost normal distribution with peak slightly to the left

    tempo = peaks seen from 100-125 and 150-160. Nothing below 75 and not much above 175.
    """
    st.markdown('<h3>Characteristics of Mainstay Songs</h3>',unsafe_allow_html=True)
    df_mainstay = df[df['is_mainstay']==1]
    df_mainstay.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'streams',
       'liveness', 'valence', 'tempo']].mean().head(10)
    #assign to new df df_mainstaytracks for EDA
    df_mainstaytracks = df_mainstay.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'streams',
       'liveness', 'valence', 'tempo']].mean()
    #Mainstay DataFrame Audio Features Distribution
    for col in ['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']:
        fig, ax = plt.subplots()
        ax = sns.distplot(df_mainstaytracks[col], kde=True)
        plt.title(col)
        plt.ylabel('Frequency')
        plt.show()
        st.pyplot(fig)
    """
    # Audio Features Observations for Mainstay Tracks:

    popularity = Mainstays are quite centered between the 60-90 range in popularity.

    danceability = Almost normally distributed. Range is between 0.36 - 0.82. Peak at 0.7 range. Mainstay tracks tend to be more danceable.

    energy = Almost normally distributed. Both low and high energy tracks can be mainstays. Peak is at 0.5.

    key = Pitch (0 = do, 2 = re, 4 = mi, 5 = fa, 7 = sol, 9 = la, 11 = ti). Well distributed.

    loudness = Skewed to the right, closely matches the distribution for all Spotify tracks. Though there is a notable dip in the -10 dB range.

    mode = (Major = 1; Minor = 0) Even more tracks are using Major (1) modality.

    speechiness = (>0.66 = purely spoken words; 0.33 - 0.66 = both music and speech such as rap; <0.33 music and non-speech tracks)
    Most values are between 0 - 0.1. 

    acousticness = (0 not acoustic; 1 acoustic) Skewed to the left, but not by much. Songs appear to be varied in terms of acousticness.

    instrumentalness = (0 has vocals; 0.5 - 1 no vocals) Majority of the tracks have vocals. In fact, no mainstay track registered a value of above 0.3

    liveness = (0 not live ; 0.8 - 1 high chance to be live) Majority of the tracks aren't live and probably taken from the album versions.

    valence = (0 sad/angry ; 1 happy/cheerful) Almost normal distribution with peak slightly to the left

    tempo = peaks seen from 125 - 140. Nothing below 75 and nothing above 162.
    """
    st.markdown('<h3>Characteristics of Ben&Ben Songs</h3>',unsafe_allow_html=True)
    df_BenandBen = df[df['artist_name']=='Ben&Ben']
    df_BenandBen.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']].mean()
    #assign to new df  for EDA
    df_BenandBentracks = df_BenandBen.groupby(['track_id','track_name','artist_name'])[['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']].mean()
    #Viral DataFrame Audio Features Distribution
    for col in ['popularity', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']:
        fig, ax = plt.subplots()
        ax = sns.distplot(df[col], kde=True)
        plt.title(col)
        plt.ylabel('Frequency')
        sns.distplot(df_BenandBentracks[col], kde=True)
        plt.show()
        st.pyplot(fig)
    """
    # Audio Features Observations for Ben and Ben Tracks:

    popularity = Ben & Ben songs are skewed to the right and are popular. Their tracks fall between the 57-71 range.

    danceability = Almost normally distributed. Good mix in terms of danceability given the range of 0.31 - 0.67.

    energy = Slightly skewed to the right. Ben&Ben has more above average energy tracks, but nothing too high.

    key = Pitch (0 = do, 2 = re, 4 = mi, 5 = fa, 7 = sol, 9 = la, 11 = ti). Well distributed.

    loudness = Slightly skewed to the right, though it is almost evenly distributed.

    mode = All tracks are using 1 (Major) modality.

    speechiness = (>0.66 = purely spoken words; 0.33 - 0.66 = both music and speech such as rap; <0.33 music and non-speech tracks)
    Most values are around the 0.3 mark, which means most of their music are still focused on instruments but there appears to be a song with a higher value of 0.485. (Make It With You)

    acousticness = (0 not acoustic; 1 acoustic) Skewed to the right, they have more tracks that are more acoustic sounding.

    instrumentalness = (0 has vocals; 0.5 - 1 no vocals) All of the tracks have vocals as there's nothing that registered above 0.005 in insturmentalness.

    liveness = (0 not live ; 0.8 - 1 high chance to be live) Majority of the tracks aren't live and probably taken from the album versions. One track registered 0.39 in liveness (Lifetime - the sounds of the starting few seconds is similar to that of a live track) Another registered 0.4 in liveness (Pagtingin, though i'm not sure why)

    valence = (0 sad/angry ; 1 happy/cheerful) Slightly skewed to the left. Majority of the songs are within 0.5 valence and below which means more of their songs can be considered sadder.

    tempo = majority within 75-150 range.
    """


# In[9]:


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


# In[10]:


if section == "Modelling for Genre Classfication":
    st.subheader('Modelling for Genre Classification')
    """
    - *Genres Considered:* Rock, Hiphop/Rap, Country, Jazz, Blues, Classical, Electronic, Folk, New Age, Reggae
    - *Features Considered:* danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo	
    - *Train Test Split Ratio:* (0.8:0.2)
    - *K Fold Cross Validation:* [4,5,8,10]
    - *Neighbors Range:* [2:50]
    """


# In[11]:


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


# In[12]:


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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Manhattan
    #get top 10 nearest to seed_track_data
    st.write('Manhattan Distribution')
    recommendation_df = chart_tracks_df[chart_tracks_df['track_id']!=seed_track_data['track_id']].sort_values('manhattan_dist')[:10]
    recommendation_df[['track_name','artist_name','manhattan_dist','predicted_genre']+feature_cols]
    chart_tracks_df['cosine_dist'] = chart_tracks_df.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),                                                                  seed_track_data[feature_cols].values.reshape(1, -1))                                                                  .flatten()[0], axis=1)
    fig, ax = plt.subplots()
    ax.hist(chart_tracks_df['manhattan_dist'], bins=20)
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Cosine
    #get top 10 nearest to seed_track_data
    st.write('Cosine Distribution')
    recommendation_df = chart_tracks_df[chart_tracks_df['track_id']!=seed_track_data['track_id']].sort_values('cosine_dist')[:10]
    recommendation_df[['track_name','artist_name','cosine_dist','predicted_genre']+feature_cols]
    
    fig, ax = plt.subplots()
    ax.hist(chart_tracks_df['cosine_dist'], bins=20)
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)


# In[13]:


if section == "Strategy":
    st.subheader("Strategy")
    """
     - Bean & Bean must collaborate with the OPM artist found by the recommender engine.
     - To broaden their audience they can choose a 2nd genre to add on their portfolio.
    """


# In[14]:


if section == "Authors":
    st.subheader("Authors")
    st.write("Eskwelabs Cohort V")
    st.markdown("<ul>"    "<li>Kenrick Nocom</li>"    "<li>Kemp Po</li>"
    "<li>Janina Elyse Reyes</li>"\
    "<li>William Raymond Revilla</li>"\
    "<li>Franz Taborlupa</li>"\
    "</ul>", unsafe_allow_html=True)

