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


section = st.sidebar.radio("Sections",
    ("Methodology","Exploratory Data Analysis","Data and Tools","Modelling for Genre Classfication","Evaluation",\
    "Recommender Engine","Strategy"))


# In[5]:


if section == "Methodology":
    st.subheader('Methodology')
    st.write('A playlist is recommended based on the results from the KNN and SVM analysis to check the accuracy of the model')


# In[6]:


if section =="Exploratory Data Analysis":
    st.subheader('Exploratory Data Analysis')


# In[7]:


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


# In[8]:


if section == "Modelling for Genre Classfication":
    st.subheader('Modelling for Genre Classification')


# In[9]:


if section == "Evaluation":
    st.subheader('Evaluation')


# In[10]:


if section == "Recommender Engine":
    st.subheader("Recommender Engine")
    title = st.text_input('Song Title', 'Maybe The Night')
    st.write('You entered the song:',title,".You might also like these songs.")
    ##Embed Spotify Playlist 
    components.html('<iframe src="https://open.spotify.com/embed/playlist/4PNDmI28zbq4GwMqO4Hyd4" width="300" height="950" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>')


# In[11]:


if section == "Strategy":
    st.subheader("Strategy")


# In[ ]:




