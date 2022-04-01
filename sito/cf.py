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
import random


"""
ritorna un dataframe MOVIE | RATING PREDETTO | GENERE (50 rows)
relativo all'utente user ordinato per rating predetto
"""
def predizioneUserWithoutMeans(user, df, df_film, rating_matrix):
    rating_matrix=rating_matrix.fillna(0)
    
    helper = df.copy()
    helper['rating'] = helper['rating'].apply(lambda x: 0 if x > 0 else 1)
    helper = helper.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)
    correlation_matrix = rating_matrix.transpose().corr(method='pearson')
    correlation_matrix[np.isnan(correlation_matrix)] = 0
    predicted_ratings = np.dot(correlation_matrix, rating_matrix)
    pd.DataFrame(predicted_ratings)
    final_predicted_ratings = np.multiply(predicted_ratings, helper)
    
    ls = final_predicted_ratings.iloc[user].sort_values(ascending = False)
    _ls=[]

    for item in ls.iteritems():
        _ls.append(item)

    ll = []
    for item in _ls:
        title=df_film.loc[df_film['movieId'] == item[0]].iloc[0]['title']
        genere=df_film.loc[df_film['movieId'] == item[0]].iloc[0]['genres']
        genere=genere.replace("|", ",")
        score = item[1]
        ll.append([title, score, genere])
        
    suggestions = pd.DataFrame(ll, columns = ['MOVIE', 'Score', 'GENERE'])
    scaler = MinMaxScaler(feature_range = (0.5, 5))
    suggestions[["RATING PREDETTO"]] = scaler.fit_transform(suggestions[["Score"]])
    del suggestions['Score']
    suggestions = suggestions[['MOVIE', 'RATING PREDETTO', 'GENERE']]
    suggestions = suggestions.head(50)

    return suggestions

"""
ritorna un dataframe MOVIE | RATING PREDETTO | GENERE (50 rows)
relativo all'utente user ordinato per rating predetto
"""
def predizioneItem(user, df, df_film, rating_matrix):
    rating_matrix=rating_matrix.fillna(0)
    
    item_similarity = cosine_similarity(rating_matrix.sub(rating_matrix.mean(),axis=0).transpose().fillna(0))
    item_similarity[np.isnan(item_similarity)] = 0
    dd=pd.DataFrame(item_similarity)
    dd.columns = rating_matrix.columns
    dd.insert(0, 'movieId', dd.columns)

    #crea lista di film non visionati dall'utente
    film_rating = []
    for index, value in rating_matrix.iloc[user-1].items():
        film_rating.append((index,value))

    film_non_visionati = list(filter(lambda item: item[1] == 0, film_rating))
    film_rated = list(filter(lambda item: item[1] != 0, film_rating))

    filmVistiEListaSimili=[]
    for (film, rate) in film_rated:
        filmVistiEListaSimili.append((film, rate, dict(list(((dd[dd['movieId'] == film]).squeeze()).items()))))
            
    #per ogni film non visionato dall'utente, seleziona top k film simili in base alla coseno similarità
    rating_stimato=[]
    for film,_ in film_non_visionati:
        den=0
        num=0
        for (_film, rate, dictSimili) in filmVistiEListaSimili:
            if(film!=_film):
                sim = dictSimili[film]
                num=num+(rate*sim)
                den=den+sim
        if(den==0):
            rating_stimato.append((film,0))
        else:
            rating_stimato.append((film,num/den))
    
    #ordina rating
    rating_stimato.sort(key = lambda x: x[1], reverse=True)
    print(rating_stimato[0])

    ll = []
    for item in rating_stimato:
        rslt_df = df_film[df_film['movieId'] == (item[0])]
        title=rslt_df.iloc[0]['title']
        genere=rslt_df.iloc[0]['genres'].replace("|", ",")
        score = item[1]
        ll.append([title, score, genere])
    
    suggestions = pd.DataFrame(ll[0:50], columns = ['TITOLO', 'RATING PREDETTO', 'GENERE'])
    return suggestions


"""
ritorna un dataframe MOVIE | RATING PREDETTO | GENERE (50 rows)
relativo all'utente user ordinato per rating predetto
"""
def predizioneUsertopK(user, df, df_film, rating_matrix):
    #salva la media di ogni utente
    rating_matrix['average'] = rating_matrix.mean(axis=1)
    media_utenti = dict()
    for index, row in rating_matrix.iterrows():
        media_utenti[index] = row['average']
    del rating_matrix['average']

    #calcola la similarità tra gli utenti
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

    #seleziona i film non visti dall'utente
    film_non_visionati=[]
    for index, value in rating_matrix.iloc[user-1].items():
        film_non_visionati.append((index,value))

    film_non_visionati = list(filter(lambda item: item[1] == 0, film_non_visionati))

    #stima il rating
    rating_stimato=[]
    for (film,_) in film_non_visionati:
        #0 id vero utente
        rate=media_utenti.get(user)
        num=0
        den=0
        for (u,similarita,lista_film_rating) in top_k:
            den=den+similarita
            rate_u_film = lista_film_rating[film] - media_utenti.get(u)
            num=num+(similarita*rate_u_film)
        
        rating_stimato.append((film,rate+(num/den)))
    rating_stimato.sort(key = lambda x: x[1], reverse=True)

    
    ll = []
    for item in rating_stimato:
        rslt_df = df_film[df_film['movieId'] == (item[0])]
        title=rslt_df.iloc[0]['title']
        genere=rslt_df.iloc[0]['genres'].replace("|", ",")
        score = item[1]
        ll.append([title, score, genere])
    
    suggestions = pd.DataFrame(ll[0:50], columns = ['TITOLO', 'RATING PREDETTO', 'GENERE'])
    return suggestions



"""
ritorna un dataframe MOVIE | RATING PREDETTO | GENERE (50 rows)
relativo all'utente user ordinato per rating predetto
"""
def predizioneUserSuperioreASoglia(user, df, df_film, rating_matrix, soglia):
    #salva la media di ogni utente
    rating_matrix['average'] = rating_matrix.mean(axis=1)
    media_utenti = dict()
    for index, row in rating_matrix.iterrows():
        media_utenti[index] = row['average']
    del rating_matrix['average']

    #calcola la similarità tra gli utenti
    rating_matrix=rating_matrix.fillna(0)
    rating_matrix_trasposta = df.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
    user_similarity = rating_matrix_trasposta.corr(method='pearson').fillna(0)
    user_similarity=pd.DataFrame(user_similarity)
    user_similarity.insert(0, 'userId', user_similarity.columns)
    rslt_df = user_similarity[user_similarity['userId'] == user]

    #seleziona quelli sopra una soglia
    lst_similarita=[]
    for index, value in (rslt_df.iloc[0]).items():
        if index != 'userId':
            lst_similarita.append((index,value,rating_matrix.iloc[index-1]))
    lst_similarita.sort(key = lambda x: x[1], reverse=True)
    top_k=lst_similarita[1:20]
    top_k=list(filter(lambda x: float(x[1])>=soglia, top_k))

    #seleziona i film non visti dall'utente
    film_non_visionati=[]
    for index, value in rating_matrix.iloc[user-1].items():
        film_non_visionati.append((index,value))

    film_non_visionati = list(filter(lambda item: item[1] == 0, film_non_visionati))

    #stima il rating
    rating_stimato=[]
    for (film,_) in film_non_visionati:
        #0 id vero utente
        rate=media_utenti.get(user)
        num=0
        den=0
        for (u,similarita,lista_film_rating) in top_k:
            den=den+similarita
            rate_u_film = lista_film_rating[film] - media_utenti.get(u)
            num=num+(similarita*rate_u_film)
        
        rating_stimato.append((film,rate+(num/den)))
    rating_stimato.sort(key = lambda x: x[1], reverse=True)

    
    ll = []
    for item in rating_stimato:
        rslt_df = df_film[df_film['movieId'] == (item[0])]
        title=rslt_df.iloc[0]['title']
        genere=rslt_df.iloc[0]['genres'].replace("|", ",")
        score = item[1]
        ll.append([title, score, genere])
    
    suggestions = pd.DataFrame(ll[0:50], columns = ['TITOLO', 'RATING PREDETTO', 'GENERE'])
    return suggestions


"""
ritorna gli utenti simili di un utente user
"""
def getUtentiSimili(user, df, rating_matrix):
    #calcola la similarità tra gli utenti
    rating_matrix=rating_matrix.fillna(0)
    rating_matrix_trasposta = df.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
    user_similarity = rating_matrix_trasposta.corr(method='pearson').fillna(0)
    user_similarity=pd.DataFrame(user_similarity)
    user_similarity.insert(0, 'userId', user_similarity.columns)

    rslt_df = user_similarity[user_similarity['userId'] == user]

    #seleziona i top k
    lst_similarita=[]
    for index, value in (list(rslt_df.iloc[0].items()))[1:]:
        lst_similarita.append((index,value, rating_matrix.iloc[index-1].items()))

    lst_similarita.sort(key = lambda x: x[1], reverse=True)
    top_k=lst_similarita[1:6]

    ll=[]
    for utente, similarita, lista_rating in top_k:
        ll.append((utente, similarita, list(filter(lambda x: x[1]!=0, lista_rating))))

    user2_=ll[0][2]
    user3_=ll[1][2]
    user4_=ll[2][2]
    user5_=ll[3][2]

    iduser2=ll[0][0]
    iduser3=ll[1][0]

    similarita2=ll[0][1]
    similarita3=ll[1][1]

    lista_finale=[]

    rating_utente = rating_matrix.iloc[user-1].items()
    rating_utente=list(filter(lambda x: x[1]!=0, rating_utente))


    for x in rating_utente:
        voto1=x[1]
        film=x[0]
        
        voto2 = list(filter(lambda y: y[0]==x[0], user2_))
        voto3 = list(filter(lambda y: y[0]==x[0], user3_))
        voto4 = list(filter(lambda y: y[0]==x[0], user4_))
        voto5 = list(filter(lambda y: y[0]==x[0], user5_))
        
        if(voto2==[]):
            voto2=0
        else:
            voto2=voto2[0][1]
            
        if(voto3==[]):
            voto3=0
        else:
            voto3=voto3[0][1]
            
        if(voto4==[]):
            voto4=0
        else:
            voto4=voto4[0][1]
            
        if(voto5==[]):
            voto5=0
        else:
            voto5=voto5[0][1]
        
        if(voto2==0 and voto3==0 and voto4 ==0 and voto5==0):
            continue
            
        lista_finale.append((film, voto1,voto2,voto3,voto4,voto5))
        
    return ((iduser2,iduser3),(similarita2, similarita3),random.sample(lista_finale, 8))