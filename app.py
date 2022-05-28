#import packages
import streamlit as st
import streamlit.components.v1 as stc
import pickle
import requests
from tmdbv3api import Movie
import pandas as pd
from tmdbv3api import TMDb
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import io
import imdb
import base64
from urllib.request import urlopen
import webbrowser
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np
from  PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve , plot_precision_recall_curve


#create_similarity function is called by rcmd(m) function to find similarity between the movies using cosine_similarity function
def create_similarity():
    data = pd.read_csv('final_final.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data,similarity


#rcmd() function is used to recommend movies to the user with the help of create_similarity function excluding the first movie returned by the function(because this is the actual movie requested itself)

def rcmd(m,n):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:n] 
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp{
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


#fetching poster from the database using api key and movie_id
def fetch_poster(imdbid):
        print("im:{}".format(imdbid))
        CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
        KEY = '35a400666209be7889bc3efc70edb67f'
        url = CONFIG_PATTERN.format(key=KEY)
        headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}
        r = requests.get(url,headers=headers)
        config = r.json()
        base_url = config['images']['base_url']
        sizes = config['images']['poster_sizes']
        IMG_PATTERN = 'http://api.themoviedb.org/3/movie/{imdbid}/images?api_key={key}' 
        r = requests.get(IMG_PATTERN.format(key=KEY,imdbid=imdbid),headers=headers)
        api_response = r.json()
        rel_path=api_response['posters'][0]['file_path']
        max_size="original"
        print(rel_path)
        url = base_url + max_size + rel_path
        posters = api_response['posters'][0]
        poster_urls = ""
        
        rel_path = posters['file_path']
        url = "{0}{1}{2}".format(base_url, max_size, rel_path)
        poster_urls+=url
        return poster_urls
def convertnum(arr,movies):
    l={}
    for i,number_of_likes in enumerate(arr):
        movie_id=movies.iloc[i].movie_id
        l[movie_id]=number_of_likes

    return l


#getting the top 6 most_liked and most_popular movie
def mostliked_and_popular(movie_list,movies,attribute):
    get_posters=[]
    get_movie_names=[]
    p=convertnum(movies[attribute],movies)
    sorted_keys = sorted(p, key=p.get,reverse=True)
    sorted_dict = {}
    tmdb = TMDb()
    tmdb.api_key = '35a400666209be7889bc3efc70edb67f'
    p=list(sorted_keys)
    for w in p[:6]:
        index = movies.index
        condition=movies['movie_id']==w
        index = index[condition]
        index=index.tolist()[0]
        movie=Movie()
        m=movie.details(w)
        poster=fetch_poster(w)
        get_posters.append(poster)
        get_movie_names+=[m.title]
    return get_movie_names,get_posters

#Getting the movie id of the selected film from imdb which should be used to get all other informations
def get_id_tmdb(m):
    tmdb = TMDb()
    tmdb.api_key = '35a400666209be7889bc3efc70edb67f'
    movie = Movie()
    search = movie.search(m)
    id=search[0].id
    movie_d=search[0]
    return id


#function to create a list of recommended movies
def get_rec_posters(rec_movies):
    rec_posters=[]
    for i in rec_movies:
        print(i)
        rec_posters.append(fetch_poster(get_id_tmdb(i)))

    return rec_posters

#getting the output(i.e.,genre of the movie) from the logistic regression model using user profile information

def predict_output_lr(x_train,x_test,y_train,gender,age,income,location,scaler):
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        pred_gb = lr.predict(x_test)
        x_new= np.array([[gender],[age],[income],[location]]).reshape(1,4)
        x_new=scaler.transform(x_new)
        return (lr.predict(x_new)[0])

#getting the output(i.e.,genre of the movie) from the GradientBoostingClassifier model using user profile information
def predict_output_gb(x_train,x_test,y_train,gender,age,income,location,scaler):
        gb = GradientBoostingClassifier()
        gb.fit(x_train, y_train)
        pred_gb = gb.predict(x_test)
        x_new= np.array([[gender],[age],[income],[location]]).reshape(1,4)
        x_new=scaler.transform(x_new)
        return (gb.predict(x_new)[0])



#getting the output(i.e.,genre of the movie) from the knn model using user profile information

def predict_output_knn(x_train,x_test,y_train,gender,age,income,location,scaler):
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)  
        pred_gb = knn.predict(x_test)
        x_new= np.array([[gender],[age],[income],[location]]).reshape(1,4)
        x_new=scaler.transform(x_new)
        return (knn.predict(x_new)[0])

def pop_sorted(movies):
    p=convertnum(movies['popularity'],movies)
    sorted_keys = sorted(p, key=p.get,reverse=True)
    tmdb = TMDb()
    tmdb.api_key = '35a400666209be7889bc3efc70edb67f'
    p=(sorted_keys)
    return sorted_keys


#getting the movie name and poster image of a movie by specifying the genre of a movie

def search_by_genre(genre,movies):
    col1,col2,col3,col4,col5,col6=st.columns(6)
    col1,col2,col3=st.columns((1,1,1))
    col4,col5,col6=st.columns((1,1,1))
    list1=pop_sorted(movies)
    c=0
    poster=list()
    title=list()
    movie=Movie()
    for i in range(len(list1)):
        
        w=int(list1[i])
        m=movie.details(w)
        if(m['genres'][0]['name']==genre ):
            title.append(m.title)
            print("title")
            print(title)
            poster.append(fetch_poster(w))
            print("title")
            print(poster)
            c+=1
        if(c==4):
            break
    with col1:
        st.write(title[0])
        st.image(poster[0])
        img=Image.open(urlopen(poster[0]))
    with col2:
        st.write(title[1])
        img=Image.open(urlopen(poster[1]))
        st.image(img)
    with col3:
        st.write(title[2])
        img=Image.open(urlopen(poster[2]))
        st.image(img)

#compare the accuracy of the three model created(logistic regression,GradientBoostingClassifier and knn) and get the highly accurated model's index
def greatest(num1,num2,num3):
    
    if ((num1 >= num2) and (num1 >= num3)):
        
        index=1
    elif ((num2 >= num1) and (num2 >= num3)):
        
        index=2
    else:
        
        index=3
    return index



movies=pd.read_pickle('movie_list.bz2')
movie=pd.read_pickle('final_final.bz2')
movie_list = movie['movie_title'].values

#setting backgroung image
set_png_as_page_bg('images\image.png')
#setting title
filmflix='<h2 style="color:#dc3545;text-align:center;font-weight:750">FilmFlix</h2>'
st.markdown(filmflix,unsafe_allow_html=True)



#Sidebar
with st.sidebar:
    choose = option_menu("Recommendations menu", ["Home","Recommendation based on similar movies","Recommendation based on user's profile", "Recommendation based on genre","Recommend based on your mood"],
                        icons=['house-fill','film','person-circle','camera-reels-fill','music-note-beamed'],
                        menu_icon="app-indicator",default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#0000"},
        "icon": {"color": "#c41610", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#7d827e"},
    }
    )


#Home page
if choose=="Home":
#sliding image in home page for ui  
    stc.html('''

  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Simple Slider using Flickity Plugin | HackerRahul.com</title>
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://unpkg.com/flickity@2/dist/flickity.min.css">
  </head>
  <style media="screen">
  .carousel-cell {
    width: 100%;
    }

    /* cell number */
    .carousel-cell:before {
      display: block;
    }
    body{
     padding:0px;
     margin:0px;

    }
  </style>
  <body>
   
    <div class="carousel" data-flickity='{ "wrapAround": true, "autoPlay": true, "imagesLoaded":true }'>
      <div class="carousel-cell">
        <img class="w3-image" src="https://www.exchange4media.com/news-photo/1524838599_r2oBUW_Picture-Avengers.jpg">
      </div>
      <div class="carousel-cell">
        <img class="w3-image" src="https://pbs.twimg.com/media/Cu8GnwpUIAABmA5.jpg:large">
      </div>
      <div class="carousel-cell">
        <img class="w3-image" src="https://www.desktopbackground.org/download/800x600/2015/09/07/1007127_spider-man-movie-posters-reflections-spiderman-3-wallpapers_1920x1080_h.jpg">
      </div>
    </div>


  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
  <script src="https://unpkg.com/flickity@2/dist/flickity.pkgd.min.js"></script>
  </body>

''')
    st.markdown("#")
    st.markdown("#")

    #Top 6 Most-liked
    if st.title('Top 6 Most liked'):
        recommended_movie_names,recommended_movie_posters = mostliked_and_popular(movie_list,movies,'number_of_likes')
        col1,col2,col3,col4,col5,col6=st.columns(6)
        col1, col2, col3 = st.columns((1,1,1))
        col4, col5,col6=st.columns((1,1,1))
    
        with col1:
            st.subheader(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.subheader(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.subheader(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        print('\n')

        with col4:
            st.subheader(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.subheader(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
        with col6:
            st.subheader(recommended_movie_names[5])
            st.image(recommended_movie_posters[5])
    #Top 6 Most-popular
    if st.title('Top 6 Most popular'):
        recommended_movie_names,recommended_movie_posters = mostliked_and_popular(movie_list,movies,'popularity')
        col1,col2,col3,col4,col5,col6=st.columns(6)
        col1, col2, col3 = st.columns((1,1,1))
        col4, col5,col6=st.columns((1,1,1))
    
        with col1:
            st.subheader(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.subheader(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.subheader(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
    
        with col4:
            st.subheader(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])

        with col5:
            st.subheader(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
        with col6:
            st.subheader(recommended_movie_names[5])
            st.markdown("#")
            st.image(recommended_movie_posters[5])



#getting gender,location,age and annual income of an user and predicting the genre of the movie the user would like to see.It also demostrates efficiency the model by plotting confusion-matrix.
if choose=="Recommendation based on user's profile":
    
    

    st.subheader("Exploratory data analysis on dataset")
    infile = open('net.pkl','rb')
    new_dict = pickle.load(infile)
    infile.close()
    scaler = StandardScaler()
    ###plots
    fig_dims = (5, 5)
    fig,ax=plt.subplots(figsize=fig_dims)
    ax.hist(new_dict['Age'], bins=20)
    st.pyplot(fig)
   

    fig_dims = (10, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.heatmap(new_dict.corr(), annot=True, cmap='inferno',ax=ax)
    st.pyplot(fig)
    

    fig,ax=plt.subplots()
    sns.countplot(new_dict['Gender'],ax=ax)
    st.pyplot(fig)


    x = new_dict.iloc[:, :-1].values
    y =new_dict.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
 

    st.subheader("Enter your details")
 
    name=st.text_input("Enter your name: ")
    option = st.selectbox(
     'What is your Gender? ',
     ('Male', 'Female'))
    if(option=="Male"):
        gender=0
    else:
        gender=1
    # gender=st.number_input("Enter your gender(0 for male/ 1 for female) : ",min_value=0,max_value=1,step=1)
    age=st.slider("Enter your age: ",min_value=0,  max_value=100)
    option = st.selectbox(
     'What is your Gender? ',
     ('Urban', 'Rural'))
    if(option=="Urban"):
        location=1
    else:
        location=0
    # location=st.number_input("Enter your location (0 for rural / 1 for urban): ",min_value=0,max_value=1,step=1)
    income=st.number_input("Enter your income in k$: ",min_value=10,max_value=10000,step=10)
    
    st.subheader("Hi {} !!!".format(name))


    #LogisticRegression model
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    pred_lr = lr.predict(x_test)
    class_names=y.unique()
    acc_1=metrics.accuracy_score(y_test, pred_lr)*100
    st.subheader("Accuracy of Logistic Regression")
    st.write(acc_1)
    st.subheader("Confusion matrix")
    plot_confusion_matrix(lr, x_test, y_test, display_labels=  class_names)
    st.pyplot()
    st.subheader("Recommendations for {} on Logistic regression".format(name))
    st.write(predict_output_lr(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler))


    
    
    #KNeighborsClassifier model
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    pred_knn = knn.predict(x_test)
    st.subheader("Accuracy of K Neighbors Classifier: ")
    acc_2=metrics.accuracy_score(y_test, pred_knn    )*100
    st.write(acc_2)
    st.subheader("Confusion matrix")
    plot_confusion_matrix(knn, x_test, y_test, display_labels=  class_names)
    st.pyplot()
    st.subheader("recommendations for {} on K Neighbors Classifier: ".format(name))
    st.write(predict_output_knn(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler))


    #GradientBoostingClassifier model
    gb = GradientBoostingClassifier()
    gb.fit(x_train, y_train)
    pred_gb = gb.predict(x_test)
    acc_3=metrics.accuracy_score(y_test, pred_gb)*100
    st.subheader("Accuracy of Gradient Boosting Classifier")
    st.write(acc_3)
    st.subheader("Confusion matrix")
    plot_confusion_matrix(gb, x_test, y_test, display_labels=  class_names)
    st.pyplot()
    st.subheader("recommendations for {} on Gradient Boosting Classifier".format(name))
    st.write(predict_output_gb(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler))


    #getting the index of highly accurate model
    index=greatest(acc_1,acc_2,acc_3)
    print("index")
    print(index)
    if(index==1):
        search_by_genre((predict_output_lr(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler)).capitalize(),movies)
    elif(index==2):
        search_by_genre((predict_output_knn(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler)).capitalize(),movies)
    else:
        search_by_genre((predict_output_gb(x_train,x_test,y_train,int(gender),int(age),int(income),int(location),scaler)).capitalize(),movies)

#filtering the movie based on genre and popularity
if choose=="Recommendation based on genre":
    st.balloons()
    st.subheader("Action movies")
    search_by_genre("Action",movies)
    st.subheader("Romance movies")
    search_by_genre("Romance",movies)
    st.subheader("Thriller movies")
    search_by_genre("Thriller",movies)
    st.subheader("Comedy movies")
    search_by_genre("Comedy",movies)
    st.subheader("Drama movies")
    search_by_genre("Drama",movies)
    


#Recomendation based on content based filtering(based on similiar films)
if(choose=="Recommendation based on similar movies"):
    
    label='<p style="color:#dc3545;">select or type the movie</p>'
    st.markdown(label,True)
    selected_movie = st.selectbox("",movie_list)
    n=st.slider("Enter the number of movies for recommendation",step=2)
    if(st.button("Recommend")):
        st.balloons()
        
        rcmd_movie=rcmd(selected_movie,n+1)
        rcmd_posters=get_rec_posters(rcmd_movie)
        cols=st.columns(n+1)
        for i in range(0,len(rcmd_movie),2):
            cols[i],cols[i+1]=st.columns((1,1))
            with cols[i]:
                st.write("\n",(rcmd_movie[i]).upper())
                st.markdown("#")
                st.markdown("#")
                st.image(rcmd_posters[i])
            with cols[i+1]:
                st.write("\n",(rcmd_movie[i+1]).upper())
                st.markdown("#")
                st.markdown("#")
                st.image(rcmd_posters[i+1])

#Recoomending music based on your mood
if choose=="Recommend based on your mood":
    st.balloons()
    lang=st.text_input("Enter the language")
    mood=st.text_input("Enter your current mood")
    singer=st.text_input("Enter your favourite singer")
    
    url="https://www.youtube.com/results?search_query={0}+{1}+song+{2}".format(lang,mood,singer)
    
    if(st.button('recommend')):
        
        webbrowser.open_new_tab(url)
                