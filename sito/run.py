from flask import Flask, flash, redirect, render_template, request, session, abort, send_from_directory

import pandas as pd
import numpy as np
from random import *
import math 
from scipy.stats import pearsonr
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cf
import asyncio
import scipy.stats

app = Flask(__name__)


#DATASET
urlRating='https://raw.githubusercontent.com/mariocuomo/progettoSII/main/ratings.csv'
urlFilm='https://raw.githubusercontent.com/mariocuomo/progettoSII/main/movies.csv'
df = pd.read_csv(urlRating)
df_film = pd.read_csv(urlFilm)

#RATING MATRIX
rating_matrix = pd.read_csv(urlRating).pivot(index = 'userId', columns = 'movieId', values = 'rating')


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/spiegazione', methods=['GET', 'POST'])
def spiegazione():
    return render_template('spiegazione.html')

@app.route("/selezionaUtente", methods=["POST"])
def selezionaUtente():
    idUtente=randint(1,610)
    
    req = request.form.get('idUtente',0)

    if req!="":
        idUtente=int(req)
    #=================
    # categorie che sono piaciute all'utente
    #================
    tipi_film = df_film['genres'].tolist()
    tipi_film= list(map(lambda st: st.replace("|", ","), tipi_film))
    st = ",".join(tipi_film)
    final = st.split(',')
    c = Counter(final)
    tipi_film=[]
    for i,s in c.items():
        tipi_film.append(i)
       
    rslt_df = df[df['userId'] == idUtente]
    merged = rslt_df.merge(df_film[['movieId', 'title', 'genres']], on = 'movieId', how = 'left')

    genres_list = merged['genres'].tolist()
    genres_list= list(map(lambda st: st.replace("|", ","), genres_list))
    st = ",".join(genres_list)
    final = st.split(',')    
    c = Counter(final)
    data=[]
    
    for i,s in c.items():
        data.append((i,s))
    tt=dict(data)

    finall=[]
    for tipo in tipi_film:
        finall.append((tipo, tt.get(tipo,0)))

    labels=[row[0] for row in finall]
    values=[row[1] for row in finall]

    #=================
    # crea dataset ['TITOLO', 'RATING', 'GENERE']
    #================
    merged = merged.drop(['userId', 'movieId', 'timestamp'], axis=1)
    merged = merged[['title', 'rating', 'genres']]
    merged['genres'] = merged['genres'].map(lambda st: st.replace("|", ","))
    merged.columns = ['TITOLO', 'RATING', 'GENERE']
    session['idUtente'] = idUtente


    #=================
    # utenti simili e relativi rating
    #================
    utentiSimili =cf.getUtentiSimili(idUtente, df, rating_matrix)
    
    film = list((x[0] for x in utentiSimili[2]))
    voti1 = list(x[1] for x in utentiSimili[2])
    voti2 = list(x[2] for x in utentiSimili[2])
    voti3 = list(x[3] for x in utentiSimili[2])

    iduser2=utentiSimili[0][0]
    iduser3=utentiSimili[0][1]

    similarita2=utentiSimili[1][0]
    similarita3=utentiSimili[1][1]
    session['maggioreSimilarita'] = similarita2


    return render_template('mostraRating.html',  tables=[merged.to_html(classes='data', index_names=False, index=False).replace('<table border="1" class="dataframe data">','<table border="1" id="customers" class="dataframe data">')], titles=merged.columns.values, utente=idUtente,
                                labels=labels, values=values,
                                film=film, voti1=voti1, voti2=voti2, voti3=voti3,
                                iduser2=iduser2, iduser3=iduser3,
                                similarita2=similarita2, similarita3=similarita3,
                                sim=round(similarita2-0.2, 1)
                                )

@app.route("/valutazione", methods=["GET", "POST"])
def valutazione():
    user = session.get('idUtente', None)

    #carica il csv
    df_film = pd.read_csv(urlFilm)
    df = pd.read_csv(urlRating)

    del df['timestamp']

    #split dei dati in train e test
    X_train, X_test = train_test_split(df, test_size = 0.2, random_state = 42)


    #=======
    # TRAIN
    #=======
    #crea rating matrix
    rating_matrix = X_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

    #matrice di correlazione tra utenti
    correlation_matrix = rating_matrix.transpose().corr(method='pearson')
    correlation_matrix[np.isnan(correlation_matrix)] = 0


    #=======
    # TEST
    #=======
    #matrice di aiuto per capire quali item sono valutati e quali no
    helper = X_test.copy()
    helper['rating'] = helper['rating'].apply(lambda x: 1 if x > 0 else 0)
    helper = helper.pivot(index ='userId', columns = 'movieId', values = 'rating').fillna(0)
    
    #crea rating matrix
    test_rating_matrix = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

    #predici il rating di ogni utente
    test_predicted_ratings = np.dot(correlation_matrix, test_rating_matrix)
    test_final_predicted_ratings = np.multiply(test_predicted_ratings, helper)
    test_final_predicted_ratings = test_final_predicted_ratings[test_final_predicted_ratings > 0]

    #scala il rating nell'intervallo 0.5 - 5 il rating di ogni utente
    scaler = MinMaxScaler(feature_range = (0.5, 5))
    scaler.fit(test_final_predicted_ratings)
    test_final_predicted_ratings = scaler.transform(test_final_predicted_ratings)


    #=======
    # VALUTAZIONE
    #=======
    #calcola il MAE
    totalRatings = np.count_nonzero(~np.isnan(test_final_predicted_ratings))
    test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')

    mae = np.abs(test_final_predicted_ratings - test).sum().sum()/totalRatings
    rmse = np.abs((test_final_predicted_ratings - test).pow(2)).sum().sum()/totalRatings
    nmae = mae/4.5

    #MOSTRA RATING PREDETTI CONTRO QUELLI VERI
    df = pd.read_csv(urlRating)
    del df['timestamp']
    
    rating_matrix = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    helper = df.copy()
    helper['rating'] = helper['rating'].apply(lambda x: 1 if x > 0 else 0)
    helper = helper.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    correlation_matrix = rating_matrix.transpose().corr(method='pearson')
    correlation_matrix[np.isnan(correlation_matrix)] = 0

    predicted_ratings = np.dot(correlation_matrix, rating_matrix)
    predicted_ratings_visti = np.multiply(predicted_ratings, helper)

    lista_film_rating_veri=[]

    for index, value in rating_matrix.iloc[user-1].items():
        if(value!=0):
            lista_film_rating_veri.append((index,value))


    lista_film_rating_predetti=[]
    _min=0
    _max=0


    for index, value in predicted_ratings_visti.iloc[user-1].items():
        if(value!=0):
            lista_film_rating_predetti.append((index,value))
            if(value<_min):
                _min=value
            if(value>_max):
                _max=value
    lista_film_rating_predetti=list(map(lambda x: (x[0],round(5*(x[1]-_min)/(_max-_min))), lista_film_rating_predetti))


    lst_return = []
    for (film,rateVero) in lista_film_rating_veri:
        ratePredetto=(list(filter(lambda x: x[0]==film, lista_film_rating_predetti)))[0][1]
        lst_return.append((film,rateVero,ratePredetto))


    ##################
    ### USER BASED
    ##################
    df = pd.read_csv(urlRating)
    rating_matrix = df.pivot(index = 'userId', columns = 'movieId', values = 'rating')

    rating_matrix['average'] = rating_matrix.mean(axis=1)
    media_utenti = dict()
    for index, row in rating_matrix.iterrows():
        media_utenti[index] = row['average']
    del rating_matrix['average']

    rating_matrix=rating_matrix.fillna(0)

    rating_matrix_trasposta = df.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
    user_similarity = rating_matrix_trasposta.corr(method='pearson').fillna(0)
    user_similarity=pd.DataFrame(user_similarity)
    user_similarity.insert(0, 'userId', user_similarity.columns)

    rslt_df = user_similarity[user_similarity['userId'] == user]
    #seleziona i top k
    lst_similarita=[]
    for index, value in (rslt_df.iloc[0]).items():
        if index != 'userId':
            lst_similarita.append((index,value,rating_matrix.iloc[index-1]))
    lst_similarita.sort(key = lambda x: x[1], reverse=True)

    top_k=lst_similarita[1:10]


    film_visionati = []
    for index, value in rating_matrix.iloc[user-1].items():
        film_visionati.append((index,value))

    film_visionati = list(filter(lambda item: item[1] != 0, film_visionati))

    rating_stimatoUser=[]
    for (film,_) in film_visionati:
        rate=media_utenti.get(user)
        num=0
        den=0
        for (u,similarita,lista_film_rating) in top_k:
            den=den+similarita
            rate_u_film = lista_film_rating[film] - media_utenti.get(u)
            num=num+(similarita*rate_u_film)        
        rating_stimatoUser.append((film,rate+(num/den)))


    ##################
    ### ITEM BASED
    ##################
    #crea rating matrix
    df = pd.read_csv(urlRating)
    rating_matrix = df.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
    rating_matrix_trasposta = df.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)

    #crea matrice di similarità
    item_similarity = cosine_similarity(rating_matrix_trasposta)
    item_similarity[np.isnan(item_similarity)] = 0
    dd=pd.DataFrame(item_similarity)
    dd.columns = rating_matrix.columns
    dd.insert(0, 'movieId', dd.columns)

    #crea lista di film visionati dall'utente
    film_rating = []
    for index, value in rating_matrix.iloc[user-1].items():
        film_rating.append((index,value))
    film_rated = list(filter(lambda item: item[1] != 0, film_rating))

    filmVistiEListaSimili=[]
    for (film, rate) in film_rated:
        filmVistiEListaSimili.append((film, rate, dict(list(((dd[dd['movieId'] == film]).squeeze()).items()))))

    #per ogni film visionato da effettuare il rate, combina i rate dei film simili
    rating_stimatoItem=[]
    for film,_ in film_rated:
        den=0
        num=0
        for (_film, rate, dictSimili) in filmVistiEListaSimili:
            if(film!=_film):
                sim = dictSimili[film]
                num=num+(rate*sim)
                den=den+sim
        if(den==0):
            rating_stimatoItem.append((film,0))
        else:
            rating_stimatoItem.append((film,num/den))

    ##################
    ### MERGE RISULTATI
    ##################
    lst_return_title=[]
    for (film, r1, r2) in lst_return:
        ratePredettoUser=(list(filter(lambda x: x[0]==film, rating_stimatoUser)))
        _ratePredettoUser=ratePredettoUser[0][1]
        ratePredettoItem=(list(filter(lambda x: x[0]==film, rating_stimatoItem)))
        _ratePredettoItem=ratePredettoItem[0][1]
        lst_return_title.append((film, r1, r2, _ratePredettoUser, _ratePredettoItem))
    print(lst_return_title)

    lst_return_title____=[]
    for (film, r1, r2,r3, r4) in lst_return_title:
        title=df_film.loc[df_film['movieId'] == film].iloc[0]['title']
        lst_return_title____.append((title, r1, r2,round(r3), round(r4)))

    return render_template('valutazione.html', mae=mae, rmse=rmse, nmae=nmae, rating=lst_return_title____, utente=user)


@app.route("/predizione", methods=["GET", "POST"])
async def predizione():
    user = session.get('idUtente', None)

    metodo="userTop10"
    
    for key, value in request.form.items():
        if key=="metodo" and value!="":
            metodo=value
    
    if metodo=='userSuperioriAK':
        mtd="USER-BASED similarità > " + str(round(session['maggioreSimilarita']-0.2, 1))
        dtf= cf.predizioneUserSuperioreASoglia(user, df, df_film, rating_matrix, session.get('maggioreSimilarita', 0))

    if metodo=='userTop10':
        mtd="USER-BASED TOP 10"
        dtf= cf.predizioneUsertopK(user, df, df_film, rating_matrix)

    if metodo=='item':
        mtd="ITEM-BASED"
        dtf= cf.predizioneItem(user, df, df_film, rating_matrix)

    if metodo=='userNoMean':
        mtd="USER-BASED WITHOUT MEAN RATING"
        dtf= cf.predizioneUserWithoutMeans(user, df, df_film, rating_matrix)

    tipi_film = df_film['genres'].tolist()
    tipi_film= list(map(lambda st: st.replace("|", ","), tipi_film))
    st = ",".join(tipi_film)
    final = st.split(',')
    c = Counter(final)
    tipi_film=[]
    for i,s in c.items():
        tipi_film.append(i)


    genres_list = dtf['GENERE'].tolist()
    genres_list= list(map(lambda st: st.replace("|", ","), genres_list))
    st = ",".join(genres_list)
    final = st.split(',')    
    c = Counter(final)
    data=[]
    
    for i,s in c.items():
        data.append((i,s))
    tt=dict(data)

    finall=[]
    for tipo in tipi_film:
        finall.append((tipo, tt.get(tipo,0)))

    labels=[row[0] for row in finall]
    values=[row[1] for row in finall]

    return render_template('predizione.html', 
                            tables=[dtf.to_html(classes='data', index_names=False, index=False).replace('<table border="1" class="dataframe data">','<table border="1" id="customers" class="dataframe data">')],
                            utente=session.get('idUtente', None), titles=dtf.columns.values,
                            labels=labels, values=values,
                            metodo=mtd,
                            sim=round(session['maggioreSimilarita']-0.2, 1))


@app.route("/nuovoUtente", methods=["GET", "POST"])
def nuovoUtente():
    rm=(rating_matrix.copy(deep=True)).fillna(0)
    #seleziona i film che hanno maggiore grado di entropia
    lst=[]
    for (colname,colval) in rm.iteritems():
        lst.append((colname,scipy.stats.entropy(colval)))
    lst.sort(key = lambda x: x[1], reverse=True)

    #arricchisci i film con i nomi
    ll = []
    for item in lst[0:50]:
        rslt_df = df_film[df_film['movieId'] == (item[0])]
        idFilm=item[0]
        title=rslt_df.iloc[0]['title']
        genere=rslt_df.iloc[0]['genres'].replace("|", ",")
        score = item[1]
        ll.append([idFilm,title, score, genere])
    #suggestions = pd.DataFrame(ll, columns = ['TITOLO', 'ENTROPIA', 'GENERE'])

    #crea vista
    return render_template('ratingNuovoUtente.html', suggestions=ll, len = len(ll))

    
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/previsioniPerTe', methods=["POST"])
def previsionePerTe():
    df_copy = df.copy()

    lst = []
    for key, value in request.form.items():
        if int(value) != 0:
            df_copy=df_copy.append({"userId": 611, "movieId": int(key), "rating":int(value), "timestamp":0}, ignore_index=True)

    rm=df_copy.pivot(index = 'userId', columns = 'movieId', values = 'rating')
    dtf= cf.predizioneItem(611, df_copy, df_film, rm)



    #======
    #   GENERI PREDETTI
    #======
    tipi_film = df_film['genres'].tolist()
    tipi_film= list(map(lambda st: st.replace("|", ","), tipi_film))
    st = ",".join(tipi_film)
    final = st.split(',')
    c = Counter(final)
    tipi_film=[]
    for i,s in c.items():
        tipi_film.append(i)

    genres_list = dtf['GENERE'].tolist()
    genres_list= list(map(lambda st: st.replace("|", ","), genres_list))
    st = ",".join(genres_list)
    final = st.split(',')    
    c = Counter(final)
    data=[]
    
    for i,s in c.items():
        data.append((i,s))
    tt=dict(data)

    finall=[]
    for tipo in tipi_film:
        finall.append((tipo, tt.get(tipo,0)))

    labelsPredette=[row[0] for row in finall]
    valuesPredette=[row[1] for row in finall]


    #======
    #   GENERI PREDETTI
    #======
    rslt_df = df_copy[df_copy['userId'] == 611]
    merged = rslt_df.merge(df_film[['movieId', 'title', 'genres']], on = 'movieId', how = 'left')

    genres_list = merged['genres'].tolist()
    genres_list= list(map(lambda st: st.replace("|", ","), genres_list))
    st = ",".join(genres_list)
    final = st.split(',')    
    c = Counter(final)
    data=[]
    
    for i,s in c.items():
        data.append((i,s))
    tt=dict(data)

    finall=[]
    for tipo in tipi_film:
        finall.append((tipo, tt.get(tipo,0)))

    valuesRated=[row[1] for row in finall]
    mtd="ITEM-BASED"


    #========
    # FILM PIU PIACIUTI
    #========
    rm=rm.fillna(0)
    l=[]
    l=[]
    for film in rating_matrix.columns:
        lst = rating_matrix[film].value_counts().items()
        voti=0
        totali=0
        for x in lst:
            if(float(x[0]>=4)):
                voti=voti+x[1]
        l.append((film,voti))
    l.sort(key = lambda x: x[1], reverse=True)

    ll = []
    for item in l[:20]:
        rslt_df = df_film[df_film['movieId'] == (item[0])]
        title=rslt_df.iloc[0]['title']
        nr = round(item[1], 4)
        genere=rslt_df.iloc[0]['genres'].replace("|", ",")
        ll.append([title, nr])
    
    return render_template('predizionePerTe.html', 
                            tables=[dtf.to_html(classes='data', index_names=False, index=False).replace('<table border="1" class="dataframe data">','<table border="1" id="customers" class="dataframe data">')],
                            utente=611, titles=dtf.columns.values,
                            labelsPredette=labelsPredette, valuesPredette=valuesPredette,
                            labelsRated=labelsPredette, valuesRated=valuesRated,
                            metodo=mtd,
                            ll=ll, len=len(ll)
                            )



app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

if __name__ == "__main__":
    app.run(debug=True, threaded=True)