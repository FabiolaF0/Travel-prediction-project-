#CODE avec les imports, le modèle choisit qui tourne.

#librairies

import streamlit as st
import pickle
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import shap
from PIL import Image

#image
img = Image.open("image.png")

#Dashboard : bouton, un titre i.e. Markdown, colonnes
st.image(img, width=700)
st.title("Travel prediction app")
st.write("""

Le taux de désengagement est défini comme la perte de clients après une certaine période. Les entreprises souhaitent cibler les clients qui sont susceptibles
de ne pas réitérer leurs commandes ou ici dans notre cas leurs voyages . Elles peuvent cibler ces clients avec des offres spéciales et des promotions
pour les inciter à rester dans l’entreprise.
Cette application prédit la probabilité qu’un client se désengage (annulation complète d’une réservation) à un mois de la réservation en utilisant les données
des clients de SELECTOUR.

""")

#Pour le côté : la date de la réservation
date = st.sidebar.date_input("Pick a date")

col1,col2=st.columns(2)
with col1:
    #1er bloc = colonne
    #Base de données clientèles uploader par les équipes markets
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

with col2:
    #2nd bloc
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(
        '''
        ### Client database ({} Customers)
        '''.format(df.shape[0])
        )
        st.dataframe(df)

#Prediction : ax.hist(X.columns) --> metrics avec F1 score etc
st.subheader('Global Performance')

# NOTE: Make sure that the class is labeled 'target' in the data file
if uploaded_file:

    df = pd.get_dummies(df, drop_first = True)

    X = df.drop('Target', axis = 1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 0)

    GBC = xgb.XGBClassifier()
    steps = [('scaler', StandardScaler()),
             ('estimator', GBC)]

    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    xgb = pipeline.score(X_test, y_test)

    col1, col2=st.columns(2)
    col1.metric(label="Projection du nombre de clients se désengageant", value="{}%".format(round(xgb*100),2))
    col2.metric(label="Nombre de clients se désengageant", value=45)

#Essaie de faire deux nouvelles colonnes
if uploaded_file:
    st.subheader("Causes")
    #3th bloc
    #Chart --> graph feature_importances_
    indices = np.argsort(GBC.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X.columns[indices][:40],x = GBC.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Importances relatives",fontsize=12)
    g.set_ylabel("Critère",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("Critères de désengagement")

        #fig, ax = plt.subplots()
        #ax.hist(X.columns)
    st.pyplot(g.figure)

#with col2:
    #Recommendation : une features = une Recommendation
st.subheader("Recommendations marketing")

if uploaded_file:
    choice = st.selectbox("Choissisez votre critère",("ServicesOpted", "AnnualLow_Income", "AnnuelMiddle_Income", "Age", "AccountSyncedToSocialMedia", "BookedHotelOrNot", "FrequentFlyer"))
    st.write('Vous avez selectionné :', choice)

    if 'ServicesOpted' in choice:
        st.write(" - Mise en place de réductions/privilèges pour des activités sur place les mieux notés.")
        st.write("- Personalisation des services avec un service à moitié prix via des notifications push.")

    if "AnnualLow_Income" in choice:
        st.write("- Réduction pour les familles nombreuses.")
        st.write("- Tarifs avantageux pour des périodes creuses dans l'année.")

    if "AnnuelMiddle_Income" in choice:
        st.write("- Mise en place de réduction pour les transferts aéroport-domicile ou aéroport-hôtel.")
        st.write("- Lors d'un choix d'une activité choisie, proposition d'une seconde activité à -20%.")

    if "Age" in choice:
        st.write(" - Campagne de mailing avec des propositions de services pour les seniors ou pour les enfants.")

    if "AccountSyncedToSocialMedia" in choice:
        st.write(" - Organisation d’un jeu concours sur les réseaux sociaux avec des hashtags.")
        st.write(" - Système de parrainage avec les comptes des réseaux sociaux.")

    if "BookedHotelOrNot" in choice:
        st.write(" - Multiplication des offres pour le choix des hôtels qui sont proposés.")
        st.write(" - Jeu concours pour gagner une nuit dans un de nos hôtels partenaires renforcer par une campagne sur les réseaux sociaux.")

    if "FrequentFlyer" in choice:
        st.write("Système de points cumulables pour les prochains voyages. Par exemple:")
        st.write(" - Si le passager a 45 ans, il peut cumuler 45 points,")
        st.write(" - points cumulables jusqu'à 100 points permettant d'avoir une réduction de -15% soit sur une activité ou une location de voiture.")
        st.write("Notification push pour choisir son siège gratuitement.")
