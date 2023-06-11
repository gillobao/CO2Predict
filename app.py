# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:55:19 2023

@author: S028171
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_shap import st_shap

# Emissions de polluants, CO2 et caractéristiques des véhicules
# commercialisés en France en 2013
df_2013 = pd.read_csv('data_2013.csv' , sep = ';', encoding='unicode_escape')
df_2013_nettoye = pd.read_csv('df_2013_nettoye.csv' , sep = ';', encoding='unicode_escape')
df = pd.read_csv('df.csv', index_col = 0)

    #---------------------------------------------------------------------------------------------------------
    #                                                    Streamlit 
    #---------------------------------------------------------------------------------------------------------

# Affichage de toutes les pages de la présentation sur toute la largeur de l'écran automatiquement:
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Sommaire du Sreamlit
## Centrer l'image en haut de la sidebar:
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image('https://www.fiches-auto.fr/sdoms/shiatsu/uploaded/part-effet-de-serre-co2-automobile-2.jpg')
 
## Affichage du titre et du plan dans la sidebar:
st.sidebar.title('Projet CO2 Predict')    
pages = ['Accueil','Introduction','Exploration et analyse des données', 
         'Modélisation : Régression multiple', 'Modélisation : Classification', 'Interprétabilité SHAP multi-classes', 
         "Prédictions : Algorithme 'CO₂ Predict'", 'Conclusion']

st.sidebar.markdown('**Sommaire**')
page = st.sidebar.radio('', pages)

## Affichage des auteurs et mentor en bas de la sidebar:
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write('### Auteurs:')
st.sidebar.write('Camille Millon')
st.sidebar.write('Gilles Ngamenye')
st.sidebar.write('Christophe Seuret')
st.sidebar.write(' ')
st.sidebar.write('### Mentor:')
st.sidebar.write('Dan Cohen')



#Chargement des datasets---------------------------------------------------

# Emissions de polluants, CO2 et caractéristiques des véhicules
# commercialisés en France en 2013
df_2013 = pd.read_csv('data_2013.csv' , sep = ';', encoding='unicode_escape')
data = pd.read_csv('data.csv', index_col = 0)
target_reg = pd.read_csv('target_reg.csv', index_col = 0)
target_reg = target_reg.squeeze()



#------------------------------------  Page 0 : accueil ----------------------------------------------------
if page == pages[0]:
    from PIL import Image
    st.image(Image.open('CO2 predict1.png'))


#------------------------------------  Page 1 : introduction ----------------------------------------------------
if page == pages[1]:
    st.write('### Introduction au projet')
    
    st.write('###### Objectifs :  \n- Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.  \n- Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/#_)')
       
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head())
    

#------------------------------------  Page 2 : exploration des données ---------------------------------------------

if page == pages[2]:
    st.write('## Exploration et analyse des données')
    

    tab1, tab2, tab3 = st.tabs(['Variables du dataset', 'Preprocessing', 'Liens entre variables'])
    
    with tab1:
 
        st.write('Deux types de variables sont disponibles : 13 qualitatives et 13 quantitatives')
        st.write('Le dataset de départ contient 44 850 lignes')
        #st.caption('Certaines variables sont redondantes (colorées de la même façon ci-dessous)')

        st.write('### Variables du dataset')
 
   
        var_num_2013 = df_2013.select_dtypes(exclude = 'object') # On récupère les variables numériques
        var_cat_2013 = df_2013.select_dtypes(include = 'object') # On récupère les variables catégorielles

        tab_num=pd.DataFrame(var_num_2013.columns,columns=['Quantitatives'])
        tab_cat=pd.DataFrame(var_cat_2013.columns,columns=['Qualitatives'])
    
    # table pour présenter les données qualitatives et quantitatives
        table1 = pd.concat([tab_num,tab_cat],axis=1).fillna('')
    
       #on définit des couleurs identiques poru les variables semblables
        def couleur1(val):
           color='white' #if val not in ('Modèle UTAC' 'Modèle dossier' 'Désignation commerciale') else 'paleturquoise'
           return 'background-color:%s' % color

     
    # code pour masquer les index
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Display a static table
        st.table(table1.style.applymap(couleur1))



    with tab2:
        
        st.write('**Etapes du preprocessing :**')
        st.write('- Suppression des doublons (619)')
        st.write('- Traitement des valeurs manquantes : concerne uniquement les variables quantitatives, remplacement par les moyennes des valeurs non manquantes')
        st.write('- Suppression des modalités sous-représentées :')
        

        fig1=px.histogram(df_2013,x="Carburant",color = 'Carburant',color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(title_text='Variable "Carburant" avant preprocessing', title_x=0.5)
        fig1.update_xaxes(categoryorder='total descending')
        
        fig2=px.histogram(df,x="Carburant",color = 'Carburant',color_discrete_sequence=px.colors.qualitative.Pastel) 
        fig2.update_layout(title_text='Variable "Carburant" après preprocessing', title_x=0.5,yaxis_range=[0,40000])
        fig2.update_xaxes(categoryorder='total descending')
        
        data_container=st.container()
        with data_container:
            commentaires,plot1, plot2 = st.columns([0.5,1,1])
            with commentaires:
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write('La variable carburant possède un grand nombre de modalités."Essence (ES)" et "Gasole (GO)" représentent plus de 99 % du portefeuille, on ne conserve donc que ces deux modalités ')
            with plot1:
                st.plotly_chart(fig1, use_container_width=True)
            with plot2:
                st.plotly_chart(fig2, use_container_width=True)

        df_2013[['boite', 'rapport']]=df_2013['Boîte de vitesse'].str.split(expand=True)
        fig3=px.histogram(df_2013,x='boite',color = 'boite',color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(title_text='Variable "boite" avant preprocessing', title_x=0.5)
        fig3.update_xaxes(categoryorder='total descending')
    
        fig4=px.histogram(df,x="boite",color = 'boite',color_discrete_sequence=['MediumTurquoise','Plum'])
        fig4.update_layout(title_text='Variable "boite" après preprocessing', title_x=0.5,yaxis_range=[0,25000])
        fig4.update_xaxes(categoryorder='total ascending')
    
        data_container=st.container()
        with data_container:
            commentaires,plot3, plot4 = st.columns([0.5,1,1])
            with commentaires:
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write('Même constat pour la variable "boite", sur laquelle seules les modalités "A" (Automatique) et "M" (Manuelle) sont conservées ')

                with plot3:
                    st.plotly_chart(fig3, use_container_width=True)
                with plot4:
                    st.plotly_chart(fig4, use_container_width=True)
  
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.write('- Sélection des variables utiles')
        

        data_container=st.container()
        with data_container:
            commentaires,image= st.columns([0.5,1])
            with commentaires:
                st.write("- Création d'une variable Cat_CO2 discrète issue de CO2 sur la base des normes suivantes:")       
            with image:
                from PIL import Image
                image0 = Image.open('etiquette-energie-voiture.jpg')
                st.image(image0,caption='',width=300)
    
        st.write('- Suppression des doublons suite aux premiers traitements (restent 5 020 lignes)')
        st.write('- La base après preprocessing est la suivante (les variables CO2 et Cat_CO2 sont les variables à expliquer):')
       #on définit des couleurs identiques poru les variables semblables

        st.dataframe(df.head())


    with tab3:
        graphe_avant,graphe_apres= st.columns([1,0.8])
        with graphe_avant:
            st.write('Matrice de corrélation avant preprocessing:')
            fig0, ax0 = plt.subplots(figsize = (3, 3))
            # get label text
            sns.set(font_scale=0.3)
            yticks, ylabels = plt.yticks()
            xticks, xlabels = plt.xticks()
            ax0.set_xticklabels(xlabels, size = 3)
            ax0.set_yticklabels(ylabels, size = 3)
            sns.heatmap(df_2013.corr(), annot = True, ax = ax0, cmap = 'magma');
            st.pyplot(fig0)
                  
            
        with graphe_apres:
            st.write('Matrice de corrélation après preprocessing:')
            fig1, ax1 = plt.subplots(figsize = (3, 3))
            sns.set(font_scale=1)
            ax1=sns.set_context("paper", rc={"font.size":8,"axes.titlesize":15,"axes.labelsize":5}) 
            sns.heatmap(df.corr(), annot = True, ax = ax1, cmap = 'magma');
            st.pyplot(fig1)
        
        #with graphe:
        #   fig, ax = plt.subplots(figsize = (2, 2))
        #   var2=df.drop(['Cat_CO2','Marque','Carburant','Carrosserie','boite','gamme2'],axis=1)
        #    st.write(var2.head())
        #    sns.pairplot(data = var2,
        #     x_vars = var2.columns,
        #     y_vars = var2.columns)
        #    plt.title('Graphique matriciel')
        #    st.pyplot(fig)

#_______________________________________________________________________________________________________
#
#                                   Page 3 : régression 
#_______________________________________________________________________________________________________

#CHARGEMENT DES LIBRAIRIES: ----------------------------------------------------------------------------
import itertools

## Les différents types de modèles de Machine Learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

## Les fonctions de paramétrage de la modélisation
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import GridSearchCV

## Les fonctions de preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

## Les fonctions statistiques
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Les métriques
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from scipy.stats import jarque_bera

## Les fonctions de sauvegarde et chargement de modèles
from joblib import dump, load


from sklearn.model_selection import train_test_split

import matplotlib as mpl





# CHARGEMENT DES JEUX DE DONNEES NETTOYES ET DES TARGETS CORRESPONDANTES: ----------------------------------------------------------------------------
data_go = pd.read_csv('data_go.csv', index_col = 0)
target_go = pd.read_csv('target_go.csv', index_col = 0)
target_go = target_go.squeeze()

data_es = pd.read_csv('data_es.csv', index_col = 0)
target_es = pd.read_csv('target_es.csv', index_col = 0)
target_es = target_es.squeeze()

df = pd.read_csv('df.csv', index_col = 0)
df.CO2 = target_reg



# FONCTIONS: ----------------------------------------------------------------------------

def standardisation_lr(data, target_reg):
    # Séparation du jeu de données en un jeu d'entrainement et de test:
    X_train, X_test, y_train, y_test = train_test_split(data, target_reg, random_state = 123, test_size = 0.2)
    
    #Standardisation des valeurs numériques + variables 'Marque' (beaucoup de catégories (>10)):
    cols = ['puiss_max', 'masse_ordma_min', 'Marque']
    sc = StandardScaler()
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])
      
    return [X_train, X_test, y_train, y_test]

def regression_lineaire(model_joblib, X_train, y_train, X_test, y_test):
    # Chargement du modèle de régression linéaire:
    lr = load(model_joblib)
    
    # Entraînement et prédictions:

    pred_train = lr.predict(X_train) # = valeurs ajustées X_train
    pred_test = lr.predict(X_test) # = valeurs ajustées X_test
    
    return [lr, pred_train, pred_test]

def selecteur(X_train, y_train, X_test, y_test):
    # Instanciation d'un modèle de régression linéaire
    lr_sfm = LinearRegression()
    
    # Création d'un sélecteur à partir de lr:
    sfm = SelectFromModel(lr_sfm)
    
    # Entrainement du selecteur et sauvegarde des colonnes de X_train sélectionnées par sfm dans sfm_train:
    sfm_train = pd.DataFrame(sfm.fit_transform(X_train, y_train), index = X_train.index)
    sfm_train = X_train[X_train.columns[sfm.get_support()]]
    
    # Sauvegarde des colonnes de X_test dans sfm_test:
    sfm_test = sfm.transform(X_test)
    sfm_test = X_test[X_test.columns[sfm.get_support()]]
        
    # Régression linéaire avec sfm_train:
    lr_sfm.fit(sfm_train, y_train)
    pred_train = lr_sfm.predict(sfm_train) # = valeurs ajustées sfm_train
    pred_test = lr_sfm.predict(sfm_test) # = valeurs ajustées sfm_test
    
    return [lr_sfm, pred_train, pred_test, sfm_train, sfm_test]

def metrics_sfm(lr_sfm, X_train, y_train, X_test, y_test, pred_train, pred_test, sfm_train, sfm_test):
    # Affichage des metrics:
    residus = pred_train - y_train 
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    x, pval = jarque_bera(residus_std)
    st.write('p_value test de normalité de Jarque-Bera: =', round(pval,2)) 
    st.write("")
    st.write("R² train =", round(lr_sfm.score(sfm_train, y_train),2))
    st.write("R² obtenu par CV =", round(cross_val_score(lr_sfm,sfm_train,y_train).mean(),2))
    st.write("R² test =", round(lr_sfm.score(sfm_test, y_test),2))
    st.write("")
    st.write('RMSE train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE test:", round(mean_absolute_error(y_test, pred_test),2))


def metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test):
    # Affichage des metrics:
    residus = pred_train - y_train 
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    x, pval = jarque_bera(residus_std)
    st.write('p_value test de normalité de Jarque-Bera: =', round(pval,2)) 
    st.write("")
    st.write("R² train =", round(lr.score(X_train, y_train),2))
    st.write("R² obtenu par CV =", round(cross_val_score(lr,X_train,y_train, cv = 5).mean(),2))
    st.write("R² test =", round(lr.score(X_test, y_test),2))
    st.write("")
    st.write('RMSE train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE test:", round(mean_absolute_error(y_test, pred_test),2))
    
def coef_lr(lr, X_train):
    # Représentation des coefficients:
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    coef = lr.coef_
    fig = plt.figure()
    plt.bar(X_train.columns, coef)
    plt.xticks(X_train.columns, rotation = 90)
    st.pyplot(fig)

def coef_sfm(lr_sfm, sfm_train):
    # Représentation des coefficients:
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    if sfm_train.shape[1] >= 1:
        fig = plt.figure()
        coef = lr_sfm.coef_
        fig = plt.figure()
        plt.bar(sfm_train.columns, coef)
        plt.xticks(sfm_train.columns, rotation = 90)
        st.pyplot(fig)

def graph_res(y_train, y_test, pred_train, pred_test):
    #Normalité des résidus:
    ## Calcul des résidus et résidus normalisés:
    residus = pred_train - y_train 
    residus_norm = (residus-residus.mean())/residus.std()
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
          
    # Graphes :
    fig = plt.figure(figsize = (15,10))
    # Espacement des graphes:
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    ## Graphe normalisation résidus:
    plt.subplot(2,2,1)
    stats.probplot(residus_norm, plot = plt)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    
    ## Graphe résidus en fonction de pred_train (valeurs ajustées):
    plt.subplot(2,2,2)
    plt.scatter(pred_train, residus, alpha = 0.3)
    plt.plot((pred_train.min(), pred_train.max()), (0, 0), lw=3, color='red')
    plt.plot((pred_train.min(), pred_train.max()), (2*residus.std(), 2*residus.std()), 'r-', lw=1.5, label = '± 2 σ') 
    plt.plot((pred_train.min(), pred_train.max()), (3*residus.std(), 3*residus.std()), 'r--', lw=1.5, label = '± 3 σ')
    plt.plot((pred_train.min(), pred_train.max()), (-2*residus.std(), -2*residus.std()), 'r-',lw=1.5)
    plt.plot((pred_train.min(), pred_train.max()), (-3*residus.std(), -3*residus.std()), 'r--', lw=1.5)
    plt.title('Résidus en fonction de pred_train (valeurs ajustées)')
    plt.xlabel('pred_train (valeurs ajustées)')
    plt.ylabel('Résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.legend(loc = 'lower left')
    
    ## Graphe boxplot des résidus:
    plt.subplot(2,2,3)
    sns.boxplot(residus, notch=True)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.title('Boite à moustache des résidus')
    plt.xlabel('résidus')
    
    ## Graphe prédictions en fonction de y_test (= le long de la droite si elles sont bonnes):
    plt.subplot(2,2,4)
    plt.scatter(pred_test, y_test, alpha = 0.3)
    plt.title('Nuage de points entre pred_test et y_test')
    plt.xlabel('pred_test')
    plt.ylabel('y_test')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), lw = 3, color ='red')
    st.pyplot(fig)
    
    
    return [residus, residus_norm, residus_std]
    
def graph_res_sfm(y_train, y_test, pred_train, pred_test):
    #Normalité des résidus:
    ## Calcul des résidus et résidus normalisés:
    residus = pred_train - y_train 
    residus_norm = (residus-residus.mean())/residus.std()
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    
    # Graphes :
    fig = plt.figure(figsize = (15,10))
    # Espacement des graphes:
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    ## Graphe normalisation résidus:
    plt.subplot(2,2,1)
    stats.probplot(residus_norm, plot = plt)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    
    ## Graphe résidus en fonction de pred_train (valeurs ajustées):
    plt.subplot(2,2,2)
    plt.scatter(pred_train, residus, alpha = 0.3)
    plt.plot((pred_train.min(), pred_train.max()), (0, 0), lw=3, color='red')
    plt.plot((pred_train.min(), pred_train.max()), (2*residus.std(), 2*residus.std()), 'r-', lw=1.5, label = '± 2 σ') 
    plt.plot((pred_train.min(), pred_train.max()), (3*residus.std(), 3*residus.std()), 'r--', lw=1.5, label = '± 3 σ')
    plt.plot((pred_train.min(), pred_train.max()), (-2*residus.std(), -2*residus.std()), 'r-',lw=1.5)
    plt.plot((pred_train.min(), pred_train.max()), (-3*residus.std(), -3*residus.std()), 'r--', lw=1.5)
    plt.title('Résidus en fonction de pred_train (valeurs ajustées)')
    plt.xlabel('pred_train (valeurs ajustées)')
    plt.ylabel('Résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.legend(loc = 'lower left')
    
    ## Graphe boxplot des résidus:
    plt.subplot(2,2,3)
    sns.boxplot(residus, notch=True)
    plt.title('Boite à moustache des résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.xlabel('résidus')
    
    ## Graphe prédictions en fonction de y_test (= le long de la droite si elles sont bonnes):
    plt.subplot(2,2,4)
    plt.scatter(pred_test, y_test, alpha = 0.3)
    plt.title('Nuage de points entre pred_test et y_test')
    plt.xlabel('pred_test')
    plt.ylabel('y_test')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), lw = 3, color ='red')
    st.pyplot(fig)
    
    return [residus, residus_norm, residus_std]

# Création d'un DataFrame regroupant les données d'origine de df enrichi des valeurs ajustées,
# des résidus, des distances de cook
def df_res(sfm_train, y_train, pred_train, residus):
    #chargement data:
    #dfdata = pd.read_csv('data.csv', index_col = 0)
    
    #Analyse statsmodel:
    X = sfm_train
    X = sm.add_constant(X) #ajout d'une constante
    y = y_train
    model = sm.OLS(y, X)
    results = model.fit()
    
    # distance de Cook (= identification des points trop influents):
    influence = results.get_influence() # results DOIT être un model de statsmodels
    (c, p) = influence.cooks_distance  # c = distance et p = p-value
    
    # AJOUT DES VARIABLES CALCULEES A DF (pred_train, résidus, résidus normalisés et distance de cook)
    
    #PRED_TRAIN:
    
    ## Création d'un DataFrame stockant pred_train en conservant les index et arrondir à une décimale:
    y_pred = pd.DataFrame(pred_train, index = sfm_train.index)
    y_pred= pd.DataFrame(y_pred.rename(columns ={0:'pred_train'}))
    y_pred = round(y_pred.pred_train,1)
   
    ## Création df1 (= Ajout de pred_train à df):
    df1 = df.join(y_pred)
    
    ## Suppression des Nans:
    df1 = df1.dropna()
    
    #RESIDUS:
    
    ## Création d'un DataFrame stockant les résidus:
    res = pd.DataFrame(residus)
    res.rename(columns ={'CO2':'residus'}, inplace = True)
    
    ## Ajout des résidus à df1:
    df1 = df1.join(res)
    
    # RESIDUS NORMALISES:
    
    ## Création d'un DataFrame stockant les résidus noramlisés:
    res_norm = pd.DataFrame(residus_norm)
    res_norm.rename(columns ={'CO2':'residus_normalisés'}, inplace = True)
    
    ## Ajout des résidus normalisés à df1:
    df1 = df1.join(res_norm)
    
    ## Labelisation des résidus normalisés à 2 écarts-types:
    liste = []
    for residus in df1.residus_normalisés:
        if residus >2 or residus <-2:
            liste.append('res norm ±2 σ')
        else:
            liste.append('ok')
    ## ajout liste (=résidus normalisés labélisés à 2 EC) à df1:
    df1['res_norm_±2_σ'] = liste
    
    ## Labelisation des résidus normalisés à 3 écarts-types:
    liste = []
    for residus in df1.residus_normalisés:
        if residus >3 or residus <-3:
            liste.append('res norm ±3 σ')
        else:
            liste.append('ok')
    ## ajout liste (=résidus normalisés labélisés à 3 EC) à df1:
    df1['res_norm_±3_σ'] = liste
    
    # DISTANCE DE COOK:
    
    ## Création d'un DataFrame stockant les distances de Cook:
    dist_cook = pd.DataFrame(c, index = res_norm.index)
    dist_cook.rename(columns ={0:'dist_cook'}, inplace = True)
    
    ## Ajout des distances de Cook à df1:
    df1 = df1.join(dist_cook)
    
    ## Labelisation des distances de Cook:
    liste = []
    for dist in df1.dist_cook:
        if dist > 4/len(y_train) or dist > 1:
            liste.append('observation influente')
        else:
            liste.append('observation non influente')
    ## ajout liste (=résidus normalisés labélisés à 3 EC) à df1:
    df1['observation_influente'] = liste
    
    # Validation des résidus élevés à 2 et 3 écarts-types:
    st.write('Pourcentage des résidus à ±2 ecarts-types (doit être <0.05) =',round(df1[(df1['residus_normalisés']>2)|(df1['residus_normalisés']<-2)].residus_normalisés.count()/df1.residus_normalisés.count(), 3))
    st.write('Pourcentage des résidus à ±3 ecarts-types (doit être <0.003) =',round(df1[(df1['residus_normalisés']>3)|(df1['residus_normalisés']<-3)].residus_normalisés.count()/df1.residus_normalisés.count(), 3))
    st.write('')

    
    # Affichage des valeurs les plus influentes du modèle:  
    fig = plt.figure(figsize = (15,5))
    ax = fig.add_subplot(111)
    ax.scatter(df1[df1['observation_influente'] == 'observation influente'].pred_train, df1[df1['observation_influente'] == 'observation influente'].residus, color = 'orange', label = 'observation influente')
    ax.scatter(df1[df1['observation_influente'] == 'observation non influente'].pred_train, df1[df1['observation_influente'] == 'observation non influente'].residus, alpha = 0.2, label = 'observation non influente')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (0, 0), lw=3, color='red')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (2*df1.residus.std(), 2*df1.residus.std()), 'r-', lw=1.5, label = '2 σ') 
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (3*df1.residus.std(), 3*df1.residus.std()), 'r--', lw=1.5, label = '3 σ')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (-2*df1.residus.std(), -2*df1.residus.std()), 'r-',lw=1.5)
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (-3*df1.residus.std(), -3*df1.residus.std()), 'r--', lw=1.5)
    ax.set(title='Résidus en fonction de pred_train (valeurs ajustées)')
    ax.set(xlabel='pred_train (valeurs ajustées)')
    ax.set(ylabel='Résidus')
    ax.legend()
    st.pyplot(fig) 
    
    st.write('')
    st.write('')
    st.write('')
    st.markdown("###### Analyse des résidus à: 👇")
    choix_EC = st.radio("",
                           ["±2 écarts-types",
                            "±3 écarts-types",
                            "résidus influant trop fortement sur le modèle (distance de Cook)"],
                           key="visibility")
    st.write('')
    st.write('')
    st.write('')

    # Représentation graphique des résidus - de quoi sont composés ces résidus élevés? - qu'est-ce qui les caractérisent?:
    
    if choix_EC != 'résidus influant trop fortement sur le modèle (distance de Cook)':
        
        if choix_EC == '±2 écarts-types':
            EC = 2
            res_boxplot = 'res_norm_±2_σ'
                
        if choix_EC == '±3 écarts-types':
            EC = 3
            res_boxplot = 'res_norm_±3_σ'
        
        st.markdown('###### Répartition des résidus élevés selon les variables catégorielles')
        
        # Représentation graphique des résidus - de quoi sont composés ces résidus élevés?:
        fig = plt.figure(figsize = (16,8))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0,
                            hspace=0.1)
        # Graphe Marque:
        plt.subplot(221)
        plt.pie(df1.Marque[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                        labels = df1.Marque[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe Carburant:
        plt.subplot(222)
        plt.pie(df1.Carrosserie[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                        labels = df1.Carrosserie[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                        autopct = lambda x: str(round(x,2))+'%',
                        labeldistance=1.2,
                        pctdistance = 0.8,
                        shadow =True)
            
        # Graphe gamme:
        plt.subplot(223)
        plt.pie(df1.gamme2[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                labels = df1.gamme2[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                autopct = lambda x: str(round(x,2))+'%',
                labeldistance=1.2,
                pctdistance = 0.8,
                shadow =True)
        
        
        # Graphe carburant:
        plt.subplot(224)
        plt.pie(df1.Carburant[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                labels = df1.Carburant[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                autopct = lambda x: str(round(x,2))+'%',
                pctdistance = 0.8,
                shadow =True)
        st.pyplot(fig)
        
        st.write('')
        st.write('')
        st.write('')
        
        st.markdown('###### Comparaison des puissance maximales et des masses en fonction de la valeur des résidus')
        
        fig = plt.figure(figsize = (10,3.5))    
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=1,
                            wspace=0.4,
                            hspace=0)
        plt.subplot(121)
        sns.boxplot(data = df1, x = res_boxplot, y = 'puiss_max')
            
        plt.subplot(122)
        sns.boxplot(data = df1, x = res_boxplot, y = 'masse_ordma_min')
        st.pyplot(fig)
        
 
    
    if choix_EC == 'résidus influant trop fortement sur le modèle (distance de Cook)':
        fig = plt.figure(figsize = (16,8))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0,
                            hspace=0.1)
        # Graphe Marque:
        plt.subplot(221)
        plt.pie(df1.Marque[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Marque[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe Carburant:
        plt.subplot(222)
        plt.pie(df1.Carrosserie[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Carrosserie[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe gamme:
        plt.subplot(223)
        plt.pie(df1.gamme2[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.gamme2[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
        
        
        # Graphe carburant:
        plt.subplot(224)
        plt.pie(df1.Carburant[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Carburant[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
        st.pyplot(fig)
        
        st.write('')
        st.write('')
        st.write('')
        
        st.markdown('###### Comparaison des puissance maximales et des masses en fonction de la valeur des résidus')
        
        fig = plt.figure(figsize = (10,3.5))    
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=1,
                            wspace=0.4,
                            hspace=0)
        plt.subplot(121)
        sns.boxplot(data = df1, x = df1['observation_influente'], y = 'puiss_max')
            
        plt.subplot(122)
        sns.boxplot(data = df1, x = df1['observation_influente'], y = 'masse_ordma_min')
        st.pyplot(fig)
    
    

# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------
if page == pages[3]:
    st.write('#### Modélisation: Régression multiple')
    
    tab1, tab2 = st.tabs(['Analyse de la variable cible CO₂', 'Régressions multiples'])
    
    with tab1:
        c1, c2 = st.columns((1,1))
        with c1:
            st.markdown("###### Choississez le type d'analyse de la variable cible CO₂ 👇")
            Analyse_Y = st.radio(" ",
                                 ["Analyse globale", "Analyse par type de carburant"],
                               key="visibility",
                               horizontal = True)
            st.write('')
            from scipy.stats import norm
            # Analyse de la distribution target (CO2 (g/km)):
            dist = pd.DataFrame(target_reg)
            
            if Analyse_Y == "Analyse globale":
                fig = plt.figure(figsize =(8,5))
                
                # Espacement des graphes:
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.2,
                                    hspace=1) 
                plt.subplot(211)
                
                # Histogramme de distribution:
                plt.hist(dist, bins=60, density=True, rwidth = 0.8, color='steelblue')
                plt.title('Histogramme de CO2 (g/km)')
                plt.xlim(0,400)
                plt.xlabel('CO2 (g/km)')
                plt.yticks([])
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                
                # Représentation de la loi normale avec la moyenne et l'écart-type de la distribution -
                # Affichage de la moyenne et la médiane de la distribution:
                
                x_axis = np.arange(0,400,1)
                plt.plot(x_axis, norm.pdf(x_axis, dist.mean(), dist.std()),'r', linewidth = 3)
                plt.xlim(0,400)
                plt.plot((dist.mean(), dist.mean()), (0, 0.015), 'r-', lw=1.5, label = 'moyenne de la distribution')
                plt.plot((dist.median(), dist.median()), (0, 0.015), 'r--', lw=1.5, label = 'médiane de la distribution')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Boite à moustache de la distribution:
                plt.subplot(212)
                sns.boxplot(x=dist.CO2, notch=True)
                plt.title('Boite à moustache de CO2 (g/km)')
                plt.xlabel('CO2 (g/km)')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.xlim(0,400)
                
                st.pyplot(fig)
                
            if Analyse_Y == "Analyse par type de carburant":
                fig = plt.figure(figsize =(10,12))
                
                # Espacement des graphes:
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.2,
                                    hspace=0.6)
                # Histogrammes de distribution des véhicules essence et des véhicules diesel :
                plt.subplot(311)
                ES = df.CO2[df['Carburant']=='ES']
                GO = df.CO2[df['Carburant']=='GO']
                
                plt.hist(ES,
                         bins=80,
                         density=True,
                         alpha=0.4,
                         color='green',
                         label ='Distribution des véhicules essence')
                
                plt.hist(GO,
                         bins=40,
                         density=True,
                         alpha=0.4,
                         color='orange',
                         label ='Distribution des véhicules diesel')
                
                plt.title('Histogramme de CO2 (g/km) en fonction du carburant')
                plt.xlabel('CO2 (g/km)')
                plt.yticks([])
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Réprésentation des distributions des véhicules essence et diesel en prenant en compte uniquement
                # leurs moyennes et leurs écarts-types (= aspect d'une loi normale avec ces moyennes et ces écarts-types):
                ## Représentation de la loi normale avec la moyenne et l'écart-type de la distribution des véhicules essence ES:
                
                plt.subplot(312)
                x_axis = np.arange(0,400,1)
                plt.plot(x_axis,
                         norm.pdf(x_axis, ES.mean(), ES.std()),
                         'g',
                         linewidth = 3,
                         alpha = 0.8,
                         label ='loi normale [ES)]')
                plt.xlim(0,400)
                plt.plot((ES.mean(), ES.mean()), (0, 0.015), 'g', lw=1.5, label = 'moyenne de la distribution ES')
                plt.plot((ES.median(), ES.median()), (0, 0.015), 'g--', lw=1.5, label = 'médiane de la distribution ES')
                
                ## Représentation de la loi normale avec la moyenne et l'écart-type de la distribution des véhicules diesel GO:
                plt.plot(x_axis,
                         norm.pdf(x_axis, GO.mean(), GO.std()),
                         'orange',
                         linewidth = 3,
                         alpha = 0.8,
                         label ='loi normale [GO]')
                plt.xlim(0,400)
                plt.plot((GO.mean(), GO.mean()), (0, 0.015), 'y', lw=1.5, label = 'moyenne de la distribution GO')
                plt.plot((GO.median(), GO.median()), (0, 0.015), 'y--', lw=1.5, label = 'médiane de la distribution GO')
                plt.title('Représentation des lois normales des distributions des véhicules essence et diesel suivant leurs moyennes et écarts-types')
                plt.xlabel('CO2 (g/km)')
                plt.yticks([])
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Boite à moustache de la distribution en fonction du carburant:
                plt.subplot(313)
                sns.boxplot(data = df, y = 'Carburant' , x = 'CO2', palette = ['green','gold'], notch=True)
                plt.xticks(rotation = 'vertical')
                plt.title('Boite à moustache de CO2 (g/km) en fonction du type de carburant')
                plt.xlabel('CO2 (g/km)')
                plt.xlim(0,400)
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                
                st.pyplot(fig)

     
    with tab2:
        st.markdown("**Méthodologie**:  \n1. sélection du dataset,  \n2. construction d'un premier modèle général à partir de l'ensemble des variables du dataset,  \n3. construction d'un second modèle affiné après sélection des variables les plus influentes,  \n3. pour chaque modèle: analyse des metrics et résidus et sélection des données les plus pertinentes, puis retour à l'étape 1")
        st.write('___')
        c1, c2, c3= st.columns((0.4, 0.4, 1))
        with c1:
            st.markdown("###### Dataset à analyser: 👇")
            choix_dataset = st.radio("",
                                     ["Dataset complet (véhicules essence et diesel)",
                                      "Véhicules diesel uniquement",
                                      "Véhicules essence uniquement"])
        
        with c2:
            st.markdown("###### Modèle de régression à analyser: 👇")
            choix_model = st.radio("",
                                   ["Modèle général",
                                    "Modèle affiné"])
        with c3:
             st.markdown("###### Analyse: 👇")
             choix_param = st.radio("",
                                    ["Metrics & Coefficients des variables",
                                     "Résidus"])


                              
        if choix_dataset == 'Dataset complet (véhicules essence et diesel)':
            dataset = data
            cible = target_reg
            model = 'lr.joblib'
           
        if choix_dataset == 'Véhicules diesel uniquement':
            dataset = data_go
            cible = target_go
            model = 'lr_go.joblib'
            
        if choix_dataset == 'Véhicules essence uniquement':
            dataset = data_es
            cible = target_es
            model = 'lr_es.joblib'
        
        if choix_model == "Modèle général":
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr, pred_train, pred_test = regression_lineaire(model, X_train, y_train, X_test, y_test)
        
        if choix_model == "Modèle affiné":
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr_sfm, pred_train, pred_test, sfm_train, sfm_test = selecteur(X_train, y_train, X_test, y_test)
        
        if choix_param == "Metrics & Coefficients des variables":
            c1, c2, c3, c4 = st.columns((1, 1.2, 0.2, 1.1))
            if choix_model == "Modèle général":
                with c1:
                    st.write("##### **Metrics:**")
                    st.write('')
                    metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test)
            
                with c2:
                    st.write("##### **Coefficients des variables:**")
                    coef_lr(lr, X_train)
            
            if choix_model == "Modèle affiné":
                with c1:
                    st.write("##### **Metrics:**")
                    st.write('')
                    metrics_sfm(lr_sfm, X_train, y_train, X_test, y_test, pred_train, pred_test, sfm_train, sfm_test)
                    
                with c2:
                    st.write("##### **Coefficients des variables retenues par le modèle:**")
                    coef_sfm(lr_sfm, sfm_train)
                
                with c4:
                    
                    if choix_dataset == 'Dataset complet (véhicules essence et diesel)':
                        st.markdown("##### Représentation graphique de la cible CO₂ par type de carburant en fonction de la masse et de la puissance des véhicules:")
                        import streamlit as st
                        from PIL import Image
                            
                        graph4D = st.radio("",
                                           ["Vidéo", "Vue 1", "Vue 2", "Vue 3", "Vue 4"],
                                           key="visibility",
                                           horizontal = True)
                        if graph4D == 'Vidéo':
                            st.video('Graphe_4D.mp4', format="video/mp4", start_time=0)
                        if graph4D == 'Vue 1':
                            image = Image.open('4D1.png')
                            st.image(image)
                        if graph4D == 'Vue 2':
                            image = Image.open('4D2.png')
                            st.image(image)
                        if graph4D == 'Vue 3':
                            image = Image.open('4D3.png')
                            st.image(image)
                        if graph4D == 'Vue 4':
                            image = Image.open('4D4.png')
                            st.image(image)
                                            
        if choix_param == "Résidus":
            c1, c2 = st.columns((1.3, 1))                
            if choix_model == "Modèle général":
                with c1:
                    st.write("##### **Analyse graphique des résidus:**")
                    residus, residus_norm, residus_std = graph_res(y_train, y_test,
                                                                   pred_train,
                                                                   pred_test)
                    st.write('')
                    st.write('')
                    st.write('')
                    
                    st.write("##### **Analyse graphique spécifique des résidus élevés et fortement influents:**")               
                    df_res(X_train, y_train, pred_train, residus)
                    
            if choix_model == "Modèle affiné":
                with c1:
                    st.write("##### **Analyse graphique des résidus:**")
                    residus, residus_norm, residus_std = graph_res_sfm(y_train, y_test,
                                                                       pred_train,
                                                                       pred_test)
                    st.write('')
                    st.write('')
                    st.write('')
                    
                    st.write("##### **Analyse graphique spécifique des résidus élevés et fortement influents:**")               
                    df_res(sfm_train, y_train, pred_train, residus)
    
        

#_______________________________________________________________________________________________________
#
#                                   Page 4 : classification 
#_______________________________________________________________________________________________________

# CHARGEMENT DES JEUX DE DONNEES NETTOYES ET DES TARGETS CORRESPONDANTES: ------------------------------
df = pd.read_csv('df.csv', index_col = 0)

## Matrice de confusion de chaque modèle:
matrix_rf = load('matrice_rf.joblib')
matrix_rf_opt = load('matrice_rf_opt.joblib')
matrix_knn = load('matrice_knn.joblib')
matrix_knn_opt = load('matrice_knn_opt.joblib')
matrix_svm = load('matrice_svm.joblib')
matrix_svm_opt = load('matrice_svm_opt.joblib')

## Rapport de classification de chaque modèle:
rap_rf = load('rapport_class_rf.joblib')
rap_rf_opt = load('rapport_class_rf_opt.joblib')
rap_knn = load('rapport_class_knn.joblib')
rap_knn_opt = load('rapport_class_knn_opt.joblib')
rap_svm = load('rapport_class_svm.joblib')
rap_svm_opt = load('rapport_class_svm_opt.joblib')
   
# On sépare les variables numériques et catégorielles
var_num = df.select_dtypes(exclude = 'object') # On récupère les variables numériques
var_cat = df.select_dtypes(include = 'object') # On récupère les variables catégorielles

# On récupère la variable cible
target_class = df['Cat_CO2'].squeeze()     # Variable cible pour la classification

var_num = var_num.drop(['CO2'], axis = 1)  # Les variables cibles sont éliminées des variables numériques
var_cat = var_cat.drop(['Cat_CO2'], axis = 1)

# Les variables catégorielles sont transformées en indicatrices
var_cat_ind = pd.get_dummies(var_cat)

# On récupère les variables explicatives
data = var_num.join(var_cat_ind)

# FONCTIONS: ----------------------------------------------------------------------------

# Fonction pour afficher les 3 matrices de confusion des 3 modèles optimisés:
def matrice(matrice, titre):
    classes = ['A','B','C','D','E','F','G']
    plt.imshow(matrice, interpolation='nearest',cmap='Blues')
    plt.title(titre)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    
    for i, j in itertools.product(range(matrice.shape[0]), range(matrice.shape[1])):
        plt.text(j, i, matrice[i, j],
                 horizontalalignment="center",
                 color="white" if (matrice[i, j] > ( matrice.max() / 2) or matrice[i, j] == 0) else "black")
    plt.ylabel('Catégories réelles')
    plt.xlabel('Catégories prédites')
    st.pyplot()

if page == pages[4]:
    st.write('#### Modélisation: Classification multi-classes')
        
    st.markdown("Explication de la démarche:  \n - un **premier modèle** est généré à partir de l'ensemble des hyperparamètres,  \n - un **second modèle optimisé** est généré après sélection des meilleurs hyperparamètres.")
    st.markdown('Nous procédons à une classification multiple. Nous avons donc choisi les classifieurs adaptés.')
    st.markdown('Nous en avons sélectionné 3 pour cette étude: SVM, KNN et Random Forest')
    tab1, tab2, tab3 = st.tabs(['Données', 'Classifications multiples', 'Comparaison des modèles'])
    
    
    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target_class,
                                                    test_size = 0.25,
                                                    random_state = 2,
                                                    stratify = target_class)

    # Les variables numériques doivent être standardisées
    cols = ['puiss_max', 'masse_ordma_min']
    sc = StandardScaler()
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])
    
    with tab1:
        st.write('#### Revue des données à classifier')
        st.markdown('Le DataFrame utilisé pour la classification multiple comporte : \n')
        c1, c2, c3 = st.columns((1.5, 1, 0.5))
        
        with c1:
            st.markdown('Les variables catégorielles :\n')
            st.write(var_cat.head())     

        
        with c2:
            st.markdown('Les variables numériques :\n')
            st.write(var_num.head())
        
        with c3:
            st.markdown('La variable cible :\n')
            st.write(target_class.head())
            
        c1, c2 = st.columns((1.5,1))
        
        with c1:
            sns.countplot(data = df, x = 'Cat_CO2', order = ('A','B','C','D','E','F','G'))
            plt.title('Répartition de la variable cible Cat_CO2')
            st.pyplot()
            
    with tab2:
        st.markdown("##### Quel modèle voulez-vous analyser? 👇")
        choix_model = st.radio(" ",
                             ["SVM",
                              "KNN",
                              "Random Forest"],
                           key="visibility",
                           horizontal = True)
        
        c1, c2 = st.columns((1.5, 1.5))
        
        
        if choix_model == 'SVM':
             rapport = rap_svm
             rapport_optim = rap_svm_opt
             matrix = matrix_svm
             matrix_optim = matrix_svm_opt
           
        if choix_model == 'KNN':
            rapport = rap_knn
            rapport_optim = rap_knn_opt
            matrix = matrix_knn
            matrix_optim = matrix_knn_opt
            
        if choix_model == 'Random Forest':            
            rapport = rap_rf
            rapport_optim = rap_rf_opt
            matrix = matrix_rf
            matrix_optim = matrix_rf_opt        
        
        fig = plt.figure(figsize = (10,10))        

        # Espacement des graphes:
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.4)
        
        with c1:
            plt.subplot(121)
            matrice(matrix, 'Matrice de confusion du modèle '+choix_model+' standard')
            st.write('Le rapport de classification '+choix_model+' standard')
            st.text(rapport) 
            
        
        with c2:
            plt.subplot(122)
            matrice(matrix_optim, 'Matrice de confusion du modèle '+choix_model+' optimisé')
            st.write('Le rapport de classification '+choix_model+' optimisé')
            st.text(rapport_optim)                 
        
        
    with tab3:
        
        # Affichage et comparaison des 3 matrices de confusion des 3 modèles optimisés
        st.write("Comparaison des matrices de confusion des 3 modèles optimisés \n")

        c1, c2, c3 = st.columns((1, 1.05, 1))
        with c1:
            # Matrice de confusion du modèle SVM optimisé:
                plt.subplot(131)
                matrice(matrix_svm_opt, 'Matrice de confusion du modèle SVM optimisé')
                st.text(rap_svm_opt)

        with c2:
            # Matrice de confusion du modèle KNN optimisé:
                plt.subplot(132)
                matrice(matrix_knn_opt, 'Matrice de confusion du modèle KNN optimisé')
                st.text(rap_knn_opt)

        with c3:
            # Matrice de confusion du modèle RF optimisé:
                plt.subplot(133)
                matrice(matrix_rf_opt, 'Matrice de confusion du modèle RF optimisé')
                st.text(rap_rf_opt)

#_______________________________________________________________________________________________________
#
#                                   Page 5 : Interprétation SHAP multi-classes 
#_______________________________________________________________________________________________________

#CHARGEMENT DES LIBRAIRIES: ----------------------------------------------------------------------------

import shap
from sklearn.tree import plot_tree

# CHARGEMENT DES MODELES: ------------------------------------------------------------------------

## Modèles:
model_rf_opt = load('clf_rf_grid.joblib')
model_knn_opt = load('clf_knn_grid.joblib')
model_svm_opt = load('clf_svc_grid.joblib')


## Explainers:
explainer_rf_opt = load('explainer_rf_opt.expected_value.joblib')
explainer_knn_opt = load('explainer_knn.expected_value.joblib')
explainer_svm_opt = load('explainer_svm_opt.expected_value.joblib')

## SHAP_VALUES:
shap_values_rf_opt = load('shap_values_rf_opt.joblib')
shap_values_knn_opt = load('shap_values_knn_opt.joblib')
shap_values_svm_opt = load('shap_values_svm_opt.joblib')



## Dataframes et Targets:
df_class = pd.read_csv('df.csv', index_col = 0)
df_class = df_class.drop('Cat_CO2', axis = 1)
df_class = df_class.drop('CO2', axis = 1)

X_test_75 = pd.read_csv('X_test_75.csv', index_col = 0)

target_class = pd.read_csv('target_class.csv', index_col = 0)
target_class = target_class.squeeze()

# Preprocessing dataset:-----------------------------------------------------------------------------
  
# On sépare les variables numériques et catégorielles
var_num = df_class.select_dtypes(exclude = 'object') # On récupère les variables numériques
var_cat = df_class.select_dtypes(include = 'object') # On récupère les variables catégorielles

# Les variables catégorielles sont transformées en indicatrices
var_cat_ind = pd.get_dummies(var_cat, drop_first = True)

# On récupère les variables explicatives
feats = var_num.join(var_cat_ind)


# Pour rappel, le dataset feats regroupe les variables numériques et catégorielles
# On répartit équitablement les classes entre les jeux d'entrainement et de test.
X_train, X_test, y_train, y_test = train_test_split(feats, target_class,
                                                    test_size = 0.25,
                                                    random_state = 2,
                                                    stratify = target_class)

# Les variables numériques doivent être standardisées
cols = ['puiss_max', 'masse_ordma_min']
sc = StandardScaler()
X_train[cols] = sc.fit_transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])

# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------
if page == pages[5]:
    st.write('#### Interprétation SHAP multi-classes')
    st.markdown("###### Modèle de classification à interpréter: 👇")
    choix_model_shap = st.radio("",
                             ["Random Forest optimisé",
                              "SVM optimisé",
                              "KNN optimisé"],
                             horizontal=False)
    
    if choix_model_shap == "Random Forest optimisé":
        model = model_rf_opt
        shap_values = shap_values_rf_opt
        matrix = matrix_rf_opt
        titre_matrix = "Matrice de confusion - Random Forest optimisé"
        explainer = explainer_rf_opt
        df1_titre = ''
        df1=''
        choix_cat = "**Random Forest optimisé**: choisir les catégories d'après la marice de confusion"
        choix_cat1 = "N'importe quel véhicule de X_test, composant la matrice de confusion, peut être analysé."
        
    if choix_model_shap == "SVM optimisé":
        model = model_svm_opt
        shap_values = shap_values_svm_opt
        matrix = matrix_svm_opt
        titre_matrix = "Matrice de confusion - SVM optimisé"
        explainer = explainer_svm_opt
        X_test = X_test_75
        y_test = y_test.loc[X_test.index]
        df1 = df.join(pd.DataFrame(model.predict(X_test), index = y_test.index))
        df1 = df1.join(df_2013['Désignation commerciale'])
        df1 = df1.rename({0:'Cat_CO2_pred'}, axis = 1)
        df1 = df1.join(pd.DataFrame(pd.DataFrame(model.predict(X_test)).index, index = y_test.index))
        df1 = df1.rename({0:'index_shape'}, axis = 1)
        df1 = df1.dropna(subset=['Cat_CO2_pred'])
        df1 = df1.rename({'CO2':'CO₂ (g/km)'}, axis = 1)
        df1 = df1[df1.columns[[8,9,3,0,10,1,2,4,5,6,7,11]]]
        df1_titre = "Tableau regroupant les véhicules pouvant être analysés pour le modèles SVM optimisé"
        choix_cat = "**SVM optimisé**: choisir le couple catégorie réelle / catégorie prédite d'après le tableau ci-dessus ☝️"
        choix_cat1 = "Afin de diminuer les temps de calcul, seuls 75 véhicules, pris au hasard dans X_test, peuvent être analysés."
        
    if choix_model_shap == "KNN optimisé":
        model = model_knn_opt
        shap_values = shap_values_knn_opt
        matrix = matrix_knn_opt
        titre_matrix = "Matrice de confusion - KNN optimisé"
        explainer = explainer_knn_opt
        X_test = X_test_75
        y_test = y_test.loc[X_test.index]
        df1 = df.join(pd.DataFrame(model.predict(X_test), index = y_test.index))
        df1 = df1.join(df_2013['Désignation commerciale'])
        df1 = df1.rename({0:'Cat_CO2_pred'}, axis = 1)
        df1 = df1.join(pd.DataFrame(pd.DataFrame(model.predict(X_test)).index, index = y_test.index))
        df1 = df1.rename({0:'index_shape'}, axis = 1)
        df1 = df1.dropna(subset=['Cat_CO2_pred'])
        df1 = df1.rename({'CO2':'CO₂ (g/km)'}, axis = 1)
        df1 = df1[df1.columns[[8,9,3,0,10,1,2,4,5,6,7,11]]]
        df1_titre = "Tableau regroupant les véhicules pouvant être analysés pour le modèle KNN optimisé"
        choix_cat = "**KNN optimisé**: choisir le couple catégorie réelle / catégorie prédite d'après le tableau ci-dessus ☝️"       
        choix_cat1 = "Afin de diminuer les temps de calcul, seuls 75 véhicules, pris au hasard dans X_test, peuvent être analysés."
        
    st.write('')
    st.write('')

    tab1, tab2 = st.tabs(['Interprétabilité globale', 'Interprétabilité locale'])
    
    with tab1:
        st.write("L'interprétabilité globale permet d'expliquer le fonctionnement du modèle de point de vue général à travers 2 graphiques:")
        
        choix_plot = st.radio("",
                              ["summary plot",
                               "dependance plot"],
                              horizontal=False)
        st.write('___')
       
        c1, c2 = st.columns((0.7, 2))
        
        with c1:
            if choix_plot == "summary plot":
                st.write("Observez, à l'aide de ces graphiques:   \n- les variables les plus importantes (ordre décroissant d'importance) et l'amplitude de leur impact sur du modèle ,   \n- l'importance des variables pour chaque catégorie.")
                st.write('')
                st.markdown("###### Summary plot à afficher: 👇")
                choix_model_shap = st.radio("",
                                            ["summary plot global",
                                             "summary plot par catégorie"],
                                            horizontal=True)
                
                if choix_model_shap == "summary plot global":
                    st.write('')
                    st.write('')
                    st.write('')
                    
                    
                if choix_model_shap == "summary plot par catégorie":
                    st.write('')
                    st.write('')
                    st.markdown("###### Catégorie à analyser: 👇")
                    choix_categorie = st.radio("",
                                               ["Catégorie A",
                                                "Catégorie B",
                                                "Catégorie C",
                                                "Catégorie D",
                                                "Catégorie E",
                                                "Catégorie F",
                                                "Catégorie G"],
                                               key = "Summary",
                                               horizontal=True)
                    
            if choix_plot == "dependance plot":
                st.write("Description de ces graphiques:   \n- l'axe des abscisses x représente la valeur d'une variable 1,   \n- l'axe des odronnées y représente les valeurs de Shapley de cette même variable 1 (une valeur de Shapley élevée tend à l'appartenance de l'observation à cette classe),   \n- les couleurs représentent la valeur d'une variable 2.")
                st.write('')
                st.write("Observez, à l'aide de ces graphiques, les valeurs des variables 1 et 2:   \n- pour une valeur de Shapley élevée (= appartenance à cette classe),   \n- pour une valeur de Shapley faible (= non-appartenance à cette classe.)")
                st.write('')
                
                            
                st.write('')
                st.markdown("###### Dependance plot à afficher: 👇")
                choix_dependance_shap = st.radio("",
                                         ["Puissance max vs Masse",
                                          "Puissance max vs Carburant",
                                        "Masse vs Carburant",
                                            "Masse vs Puissance max"],
                                         horizontal=False)  
                st.write('')
                st.write('')
                
                st.markdown("###### Catégorie à analyser: 👇")
                choix_categorie = st.radio("",
                                           ["Catégorie A",
                                            "Catégorie B",
                                            "Catégorie C",
                                            "Catégorie D",
                                            "Catégorie E",
                                            "Catégorie F",
                                            "Catégorie G"],
                                           key = "Dependance",
                                           horizontal=True)
                
        with c2: 
            if choix_plot == "summary plot":
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                if choix_model_shap == "summary plot global":
                    # Summary_plot:
                        st_shap(shap.summary_plot(shap_values,
                                                  X_test,
                                                  plot_type="bar",
                                                  class_names = ['A', 'B', 'C', 'D', 'E', 'F','G']))
                                    
                if choix_model_shap == "summary plot par catégorie":                
                    if choix_categorie == "Catégorie A":
                        st_shap(shap.summary_plot(shap_values[0],
                                                  X_test,
                                                  feature_names=feats.columns))
                                        
                    if choix_categorie == "Catégorie B":
                        st_shap(shap.summary_plot(shap_values[1],
                                                  X_test,
                                                  feature_names=feats.columns))
                                            
                    if choix_categorie == "Catégorie C":
                        st_shap(shap.summary_plot(shap_values[2],
                                                  X_test,
                                                  feature_names=feats.columns))
                                        
                    if choix_categorie == "Catégorie D":
                        st_shap(shap.summary_plot(shap_values[3],
                                                  X_test,
                                                  feature_names=feats.columns))
                                        
                    if choix_categorie == "Catégorie E":
                        st_shap(shap.summary_plot(shap_values[4],
                                                  X_test,
                                                  feature_names=feats.columns))
                                        
                    if choix_categorie == "Catégorie F":
                        st_shap(shap.summary_plot(shap_values[5],
                                                  X_test,
                                                  feature_names=feats.columns))
                                        
                    if choix_categorie == "Catégorie G":
                        st_shap(shap.summary_plot(shap_values[6],
                                                  X_test,
                                                  feature_names=feats.columns))
                        
            if choix_plot == "dependance plot":        
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                st.write('')
                
                if choix_dependance_shap == "Puissance max vs Masse":
                    if choix_categorie == "Catégorie A":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[0], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))                       
                                                    
                    if choix_categorie == "Catégorie B":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[1], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))   
                        
                    if choix_categorie == "Catégorie C":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[2], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))       
                        
                    if choix_categorie == "Catégorie D":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[3], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min")) 
                        
                    if choix_categorie == "Catégorie E":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[4], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))
                        
                    if choix_categorie == "Catégorie F":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[5], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))
                        
                    if choix_categorie == "Catégorie G":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[6], 
                                                     X_test, 
                                                     interaction_index= "masse_ordma_min"))
                
                
                if choix_dependance_shap == "Puissance max vs Carburant":
                    if choix_categorie == "Catégorie A":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[0], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))                       
                                                    
                    if choix_categorie == "Catégorie B":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[1], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))   
                        
                    if choix_categorie == "Catégorie C":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[2], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))       
                        
                    if choix_categorie == "Catégorie D":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[3], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))   
                        
                    if choix_categorie == "Catégorie E":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[4], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))
                                        
                    if choix_categorie == "Catégorie F":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[5], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO")) 
                        
                    if choix_categorie == "Catégorie G":
                        st_shap(shap.dependence_plot("puiss_max", 
                                                     shap_values[6], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))
                        
                if choix_dependance_shap == "Masse vs Carburant":
                    if choix_categorie == "Catégorie A":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[0], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))                       
                                                    
                    if choix_categorie == "Catégorie B":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[1], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))   
                        
                    if choix_categorie == "Catégorie C":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[2], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))       
                        
                    if choix_categorie == "Catégorie D":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[3], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))   
                        
                    if choix_categorie == "Catégorie E":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[4], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO"))
                                        
                    if choix_categorie == "Catégorie F":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[5], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO")) 
                        
                    if choix_categorie == "Catégorie G":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[6], 
                                                     X_test, 
                                                     interaction_index= "Carburant_GO")) 
                        
                if choix_dependance_shap == "Masse vs Puissance max":
                    if choix_categorie == "Catégorie A":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[0], 
                                                     X_test, 
                                                     interaction_index= "puiss_max"))
                    if choix_categorie == "Catégorie B":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[1], 
                                                     X_test, 
                                                     interaction_index= "puiss_max")) 
                    if choix_categorie == "Catégorie C":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[2], 
                                                     X_test, 
                                                     interaction_index= "puiss_max")) 
                    if choix_categorie == "Catégorie D":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[3], 
                                                     X_test, 
                                                     interaction_index= "puiss_max")) 
                    if choix_categorie == "Catégorie E":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[4], 
                                                     X_test, 
                                                     interaction_index= "puiss_max")) 
                    if choix_categorie == "Catégorie F":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[5], 
                                                     X_test, 
                                                     interaction_index= "puiss_max")) 
                    if choix_categorie == "Catégorie G":
                        st_shap(shap.dependence_plot("masse_ordma_min", 
                                                     shap_values[6], 
                                                     X_test, 
                                                     interaction_index= "puiss_max"))
               
    with tab2:
        c1, c2  = st.columns((1, 0.1))
        with c1:
            st.write("L'interprétabilité locale permet d'expliquer le fonctionnement du modèle pour une instance.")
            st.write('')
        
        c1, c2, c3  = st.columns((0.4, 0.1, 1.6))
        
        with c1:
            plt.subplot(111)
            matrice(matrix, titre_matrix)
                        
            
        with c3:
            st.write(df1_titre)
            st.write(df1)
        
        c1, c2  = st.columns((1, 0.1))
        with c1:
            st.write('___')
        
        c1, c2  = st.columns((1, 0.1))
        with c1:
            
            st.write("###### Choisir les catégories réelle et prédite:  \n-", choix_cat, "  \n-", choix_cat1)
            st.write('')
            st.write('')
            
        c1, c2, c3 = st.columns((0.25, 0.25, 1))
        with c1:
            st.markdown("###### Catégorie réelle: 👇")
            choix_cat_reel = st.radio("",
                                      ["A",
                                       "B",
                                       "C",
                                       "D",
                                       "E",
                                       "F",
                                       "G"],
                                      horizontal=False)
        
        with c2:    
            st.markdown("###### Catégorie prédite: 👇")
            choix_cat_pred = st.radio(" ",
                                      ["A",
                                       "B",
                                       "C",
                                       "D",
                                       "E",
                                       "F",
                                       "G"],
                                      horizontal=False)
        c1, c2  = st.columns((1, 1))
        with c1:
            st.write('')
            st.write('')
            # Création d'un DataFrame regroupant par index (véhicules) les catégories réelles de pollution, les catégories prédites
            # le modèle et l'index de y_pred
            
            df2 = df.join(pd.DataFrame(model.predict(X_test), index = y_test.index))
            df2 = df2.join(df_2013['Désignation commerciale'])
            df2 = df2.rename({0:'Cat_CO2_pred'}, axis = 1)
            df2 = df2[(df2['Cat_CO2'] == choix_cat_reel)&(df2['Cat_CO2_pred'] == choix_cat_pred)]
            df2 = df2.join(pd.DataFrame(pd.DataFrame(model.predict(X_test)).index, index = y_test.index))
            df2 = df2.rename({0:'index_shape'}, axis = 1)
            df2 = df2.rename({'CO2':'CO₂ (g/km)'}, axis = 1)
            df2 = df2.dropna(subset=['Cat_CO2_pred'])
            df2 = df2[df2.columns[[8,9,3,0,10,1,2,4,5,6,7,11]]]
                        
              
            
            st.dataframe(df2)
                
            st.write('')
            st.markdown("###### D'après le tableau ci-dessus ☝️, choisir l'index du véhicule à analyser: 👇")
            index = st.selectbox("",
                                 df2.index)
                                    
            if index == None:
                st.markdown("###### Aucun véhicule ne correspond à votre choix - Tous les véhicules n'ont pas pu être analysés avec ce modèle de classification  \n###### Veuillez vous référer au premier tableau, à droite de la matrice de confusion, pour choisir les bonnes catégories")
            
            else:
                st.write('')
                st.write('')
                st.write("###### Vous avez choisi d'analyser ce véhicule:")
                st.dataframe(df2[df2.index == index])
                st.write('')
                st.write('')
                st.write("Observez, à l'aide de ces graphiques:   \n- quelles variables ont un impact positif (rouge) ou négatif (bleu) sur la prédiction d'appartenance à une classe,   \n- l'amplitude de cet impact.")
                st.write('')
                st.write('')
                j=df2.loc[index].index_shape
                k = 0
                liste = ['Catégorie A', 'Catégorie B', 'Catégorie C', 'Catégorie D','Catégorie E','Catégorie F','Catégorie G']
                for k in range(0,7,1):
                    st.caption(liste[k])
                    st_shap(shap.force_plot(explainer[k], shap_values[k][j,:], X_test.iloc[j,:]))
                    k = k+1


#_______________________________________________________________________________________________________
#
#                                   Page 6 : Prédictions: Algorithme 'CO₂ Predict' 
#_______________________________________________________________________________________________________

# Chargement des modèles:
lr_es = load('lr_es.joblib')
lr_go = load('lr_go.joblib')
sfm_es = load('sfm_es.joblib')
sfm_go = load('sfm_go.joblib')


#explainer:
explainer_rf_opt = load('explainer_rf_opt2.joblib')
#explainer_svm_opt = load('explainer_svm.joblib')
explainer_knn_opt = load('explainer_knn_opt.joblib')

#explainer_expected_values:
explainer_rf_opt_exp_val = load('explainer_rf_opt.expected_value.joblib')
explainer_knn_opt_exp_val = load('explainer_knn.expected_value.joblib')
explainer_svm_opt_exp_val = load('explainer_svm_opt.expected_value.joblib')


# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------

if page == pages[6]:
    st.write("#### Prédictions: Algorithme 'CO₂ Predict'")
    st.markdown("- Utilisez notre algorithme **'CO₂ Predict'** pour prédire les rejets de CO₂ et la catégorie de pollution de votre véhicule.  \n- Les algoritmes de régression et de classification étant différents, il se peut qu'une prévision de rejets de CO₂ par régression ne correspondent pas à la catégorie d'émission prédite par un algoritme de classification.  \n- Choisissez la valeur de chaque variable avec cohérence et prenez du recul sur l'interprétation.")
    st.write('___')
    st.write("###### Configurez votre véhicule: 👇")
    st.write('')


    c1, c2, c3 = st.columns((1,1,1))
    with c1:
        marque = st.selectbox("Marque:", df.Marque.unique())
        
        carburant = st.selectbox("Carburant:",  ["Essence", "Diesel"])
        if carburant == "Essence":
            carburant = "ES"
            
        else:
            carburant = "GO"
            
        carrosserie = st.selectbox("Carrosserie:", df.Carrosserie.unique())
        
    with c2:
        # Données:       
        puissance = st.slider('Puissance (CV):', 40, 540, value = 150)
        
        masse = st.slider('Masse (kg):', 900, 3000, value = 1500)  
        
        

        
    with c3:
        boite = st.selectbox("Boite:", ["Manuelle", "Automatique"])
        if boite == "Manuelle":
            boite = "M"
            
        else:
            boite = "A"
            
        gamme = st.selectbox("Gamme:", df.gamme2.unique())
        
        #Création du dataframe avec ces nouvelles données:    
        dic = {'Marque': marque,
               'Carburant': carburant,
               'puiss_max': puissance, 
               'masse_ordma_min': masse,
               'Carrosserie': carrosserie,
               'boite': boite,
               'gamme2': gamme}
                    
        new_car = pd.DataFrame(data = dic, index = ['0'])
        
        new_df = df_class.append(dic, ignore_index=True)
        
    st.write(' ')
    if carburant == "ES":
        st.write("L'algorithme a été entrainé sur un dataset dont les limites pour les variables 'Puissance' et 'Masse' des véhicules essence sont:   \n- 44 CV < puissance < 540 CV,   \n- 825 kg < masse < 3115 kg")
    
    if carburant == "GO":
        st.write("L'algorithme a été entrainé sur un dataset dont les limites pour les variables 'Puissance' et 'Masse' des véhicules diesel sont:   \n- 40 CV < puissance < 280 CV,   \n- 845 kg < masse < 2680 kg")

  
    st.write('___')


    reg_predict, classif_predict, SHAP = st.columns((0.45,0.45,1.1))
    with reg_predict:
        #préproceesing régression:  
        dic_reg = {'Marque': marque,
                   'puiss_max': puissance, 
                   'masse_ordma_min': masse,
                   'Carrosserie': carrosserie,
                   'boite': boite,
                   'gamme2': gamme}
        
        if carburant == 'ES':
            new_car = new_car.drop(['Carburant'], axis = 1)
            df_ES = df[df.Carburant =='ES']
            df_ES = df_ES.drop(['Carburant', 'CO2', 'Cat_CO2'], axis = 1)
            df_ES = df_ES.append(dic_reg, ignore_index=True)
            
                
        # On sépare les variables numériques et catégorielles
            var_num_new = df_ES.select_dtypes(exclude = 'object') # On récupère les variables numériques
            var_cat_new = df_ES.select_dtypes(include = 'object') # On récupère les variables catégorielles 
        
        # Labélisation des variables catégorielles par labelencoder:
            labelencoder = LabelEncoder()
            var_cat_num = var_cat_new.apply(labelencoder.fit_transform)
        
            data_ES = var_num_new.join(var_cat_num)
            
            new_car_num = data_ES.loc[[2057]]
            data_num = data_ES.drop([2057], axis = 0)    
                
            target = pd.DataFrame(target_es, index = data_num.index)
        
        else:
            new_car = new_car.drop(['Carburant'], axis = 1)
            df_GO = df[df.Carburant =='GO']
            df_GO = df_GO.drop(['Carburant', 'CO2', 'Cat_CO2'], axis = 1)
            df_GO = df_GO.append(dic_reg, ignore_index=True)
                        
            # On sépare les variables numériques et catégorielles
            var_num_new = df_GO.select_dtypes(exclude = 'object') # On récupère les variables numériques
            var_cat_new = df_GO.select_dtypes(include = 'object') # On récupère les variables catégorielles 
            
            # Labélisation des variables catégorielles par labelencoder:
            labelencoder = LabelEncoder()
            var_cat_num = var_cat_new.apply(labelencoder.fit_transform)
            
            data_GO = var_num_new.join(var_cat_num)
            
            new_car_num = data_GO.loc[[2961]]
            data_num = data_GO.drop([2961], axis = 0)    
                    
            target = pd.DataFrame(target_go, index = data_num.index)
            
            
        X_train, X_test, y_train, y_test = train_test_split(data_num, target, random_state = 123, test_size = 0.2)
    
        #Standardisation des valeurs numériques + variables 'Marque' (beaucoup de catégories (>10)):
        cols = ['puiss_max', 'masse_ordma_min', 'Marque']
        sc = StandardScaler()
        X_train[cols] = sc.fit_transform(X_train[cols])
        new_car_num[cols] = sc.transform(new_car_num[cols])
    

    
       
        if carburant == 'ES':
            st.markdown("###### Sélectionnez l'algorithme: 👇")
            choix_lr_pred = st.radio(" ",
                                     ["Modèle général Essence",
                                      "Modèle affiné Essence"],
                                     horizontal=False)
            
            if choix_lr_pred == "Modèle général Essence":
                model = lr_es
            if choix_lr_pred == "Modèle affiné Essence":
                model = sfm_es
                new_car_num = new_car_num[['puiss_max','masse_ordma_min']]
        
        else:
            st.markdown("###### Sélectionnez l'algorithme de régression: 👇")
            choix_lr_pred_go = st.radio(" ",
                                        ["Modèle général Diesel",
                                         "Modèle affiné Diesel"],
                                        horizontal=False)

            if choix_lr_pred_go == "Modèle général Diesel":
                model = lr_go
            if choix_lr_pred_go == "Modèle affiné Diesel":
                model = sfm_go
                new_car_num = new_car_num[['masse_ordma_min']]
    
        new_car_pred_lr = model.predict(new_car_num)
        pred_CO2 = new_car_pred_lr[0]
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('Prédictions des rejets de CO₂ (en g/km):') 
        st.subheader(np.round(pred_CO2,0))    
        
        
        
    with classif_predict:
        #préproceesing classification:    
        # On sépare les variables numériques et catégorielles
        var_num_new = new_df.select_dtypes(exclude = 'object') # On récupère les variables numériques
        var_cat_new = new_df.select_dtypes(include = 'object') # On récupère les variables catégorielles
        
        var_cat_num1 = pd.get_dummies(var_cat_new, drop_first = True)
            
        new_df_enc = var_num_new.join(var_cat_num1)
        
        
        new_car_enc = new_df_enc.loc[[5018]]
        new_df_enc = new_df_enc.drop([5018], axis = 0)
        
        X_train, X_test, y_train, y_test = train_test_split(new_df_enc, target_class,
                                                            test_size = 0.25,
                                                            random_state = 2,
                                                            stratify = target_class)
        # Les variables numériques doivent être standardisées
        cols = ['puiss_max', 'masse_ordma_min']
        sc = StandardScaler()
        X_train[cols] = sc.fit_transform(X_train[cols])
        new_car_enc[cols] = sc.transform(new_car_enc[cols])
        
        #Prédictions:
        st.markdown("###### Sélectionnez l'algorithme de classification: 👇")
        choix_model_pred = st.radio("",
                                    ["Random Forest optimisé (= le meilleur)",
                                     "SVM optimisé",
                                     "KNN optimisé"],
                                    horizontal=False)
        
        if choix_model_pred == "Random Forest optimisé (= le meilleur)":
            model = model_rf_opt
            explainer = explainer_rf_opt
            expected_values = explainer_rf_opt_exp_val
            message = "###### Analysez les graphiques suivants pour comprendre les raisons ayant poussé 'CO₂ Predict' à classer votre véhicule dans cette catégorie: 👇"
        if choix_model_pred == "SVM optimisé":
            model = model_svm_opt
            message = "A cause de temps de calcul trop longs, il n'est pas possible d'afficher les force_plots du modèle SVM via cette application."
            expected_values = explainer_svm_opt_exp_val
        if choix_model_pred == "KNN optimisé":
            model = model_knn_opt
            explainer = explainer_knn_opt
            expected_values = explainer_knn_opt_exp_val
            message = "###### Analysez les graphiques suivants pour comprendre les raisons ayant poussé 'CO₂ Predict' à classer votre véhicule dans cette catégorie: 👇"
    
        new_car_pred_cat = model.predict(new_car_enc)
        pred_CO2_cat = new_car_pred_cat[0]
        st.write('')
        st.write('')
        st.write('')
        st.write("Prédiction de la catégorie d'émission de CO₂:")
        st.subheader(pred_CO2_cat)
        
    
        from PIL import Image
        image_pred = Image.open('etiquette-energie-voiture.jpg')
        st.image(image_pred,caption='', width=300)
        
    with SHAP:
        
        
        st.write(message)
        st.write('')
        st.write('')
        if choix_model_pred == "Random Forest optimisé (= le meilleur)" or choix_model_pred == "KNN optimisé":
            shap_values = explainer.shap_values(new_car_enc)
            k = 0
            liste = ['Catégorie A', 'Catégorie B', 'Catégorie C', 'Catégorie D','Catégorie E','Catégorie F','Catégorie G']
            for k in range(0,7,1):
                st.caption(liste[k])
                st_shap(shap.force_plot(expected_values[k], shap_values[k][0], new_car_enc.iloc[0,:]))
                k = k+1




# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------

if page == pages[7]:
    st.write("#### Conclusion")
    st.write("Les algorithmes, aussi puissants soient-ils, ne nous donnent qu’un résultat de prédiction, ce qui suscite beaucoup de questions sur leurs utilisations (éthique, juridique, bonne prise de décision, etc…). Comment avoir réellement confiance en ces prédictions ? Les méthodes d’interprétabilité et d’explicabilité de ces modèles, telles que SHAP ou LIME, répondent, en partie, à ces interrogations.  Elles apportent confiance et transparence. Les modèles de Machine Learning (ML) et Deep Learning (DL) sont souvent décrits comme des « boîtes noires ». Ces méthodes allument la lumière de ces boîtes noires. Comprendre le fonctionnement d’un modèle dans sa globalité et les causes d’une prédiction constituent une étape cruciale dans l’acceptation, le déploiement, l’utilisation, la connaissance des limites  des modèles de ML et DL. L’interprétabilité apporte du sens à la modélisation.")
    st.write("Cependant, ces outils ont aussi des inconvénients. Pour la méthode SHAP utilisée dans ce projet, le temps de calcul de cet algorithme sur le jeu de test (1255 observations – 66  variables) pouvait atteindre plus de 20h pour les modèles SVM et KNN!  Cette contrainte a été compensée en échantillonnant 75 observations, dégradant malheureusement la qualité de l’interprétabilité. De plus, l’interprétabilité d’une nouvelle observation nous oblige à relancer ces calculs, couteux en temps et énergie.")
    st.write("Bien que la méthode SHAP constitue une aide indispensable à la prise de décision, celle-ci n’échappe pas aux compromis.")
    st.write("Plus généralement, il faut prêter une attention particulière à la valeur de chaque variable choisie pour calculer une prédiction. Un modèle calculera, affichera une prédiction quelles que soient ses entrées, vous montrera les graphiques d’interprétabilité. Il est par exemple possible de calculer une prédiction de rejets de CO2 pour une Bentley, minibus, de 70 CV, 2900 kg, à moteur essence, en boite manuelle et de gamme luxe. Même si l’évocation de ce véhicule prête à sourire, ‘CO2 Predict’ calcule les émissions, la catégorie de pollution et vous donne les raisons de ce classement. Or, ce type de véhicule n’a aucun sens. La responsabilité de l’utilisateur tient notamment dans la cohérence des valeurs de chaque variable. Le résultat d’une prédiction n’exclut pas le bon sens ! La place de l’humain reste centrale dans cette univers ‘data’.")
    st.write("Ce projet de prédiction des rejets de CO2 nous a permis de mettre moins en avant une qualité de prédiction par l’utilisation de modèles de ML qu’une présentation, non exhaustive, de l’utilité de l’interprétabilité de ces modèles.")

