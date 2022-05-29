---
title: FilmFlix
emoji: 📈
colorFrom: gray
colorTo: red
sdk: streamlit
sdk_version: 1.9.0
app_file: app.py
pinned: false
---

# FilmFlix

Link to the app: https://filmflix-0.herokuapp.com/

## Overview 
   ### Home
   It contains posters of Top 6 most liked and most popular movies.These movies are sorted based on number of like and popularity measures.
       
   ### Recommendation based on similar movies:
   This works based on content based recommender system using cosine similarity.Cosine similarity is a metric used to measure how similar two items are.
       
   ### Recommendation based on user profile:
   This section first performs exploratory data analysis on the dataset and some useful plots are displayed. 
   
   Then it builds three ml models:
   
   *  KNearestNeigbour
   
   *  GradientBoostingClassifier
   
   *  LogisticRegression
   
   Then it compares the accuracy of three models and suggests genre of the movie based on high accuracy.
             
   ### Recommendation based on Genre
   This section searches and filters the five genres of the movie.
   *  Action
   *  Romance
   *  Thriller
   *  Comedy
   *  Drama
            
 ### Recommendation based on your mood
 This section recommends music based on language, mood and favourite singer of the user.
       
 ## TechStack
 
 1.Python
 
 2.Streamlit
 
 3.TMDb api
 
 4.Html (inside streamlit.components.v1)
          
## How to run the project?

1.Clone or download this repository to your local machine.

2.Then install the virtal environment in the file directory using the command:
  `python -m venv my_venv`.

3.Activate the virtual environment by the placing the relative path of Activate.ps1 in my_venv:
  `my_venv\Scripts\Activate.ps1`

4.`pip install -r requirements.txt`

5.`streamlit run app.py`

## ScreenShots
![Home](https://github.com/SAPARNA-K/FilmFlix/blob/master/screenshots/Home.png)
!(https://github.com/SAPARNA-K/FilmFlix/blob/master/screenshots/Recommendation%20based%20on%20similar%20movies%202.png)

              
              
              
       
       
   
   
