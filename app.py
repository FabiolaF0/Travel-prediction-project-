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

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import shap
from PIL import Image

#image
img = Image.open("image.png")

#code

df = pd.read_csv('Customertravel.csv')
df.rename(columns={'Target': 'Churn'}, inplace=True)

#Variable catégorielle avec Map

df["FrequentFlyer"] = df['FrequentFlyer'].map({"Yes": 0, "No": 1, "No Record": 2})
df["AccountSyncedToSocialMedia"] = df['AccountSyncedToSocialMedia'].map({"Yes": 0, "No": 1})
df["BookedHotelOrNot"] = df['BookedHotelOrNot'].map({"Yes": 0, "No": 1})

df = pd.get_dummies(df, columns=['AnnualIncomeClass'],drop_first = True)
qualitative_cols = [
        'FrequentFlyer',
        'AnnualIncomeClass',
        'AccountSyncedToSocialMedia',
        'BookedHotelOrNot'
        ]
quantitative_cols = [
        'Age',
        'ServicesOpted'
        ]
target_col = ['Churn']

df_scale = df

scaler = StandardScaler()
df_scale[quantitative_cols]= scaler.fit_transform(df_scale[quantitative_cols])

X = df.drop('Churn', axis = 1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 0)

GBC1 = xgb.XGBClassifier()
GBC1.fit(X_train, y_train)
y_pred = GBC1.predict(X_test)
xgb = GBC1.score(X_test,y_test)

feat_importances = pd.Series(GBC1.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh', color='C0')

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

st.subheader('Performances globales')

if uploaded_file:
    col1, col2=st.columns(2)
    col1.metric(label="Projection du nombre de clients se désengageant", value="{}%".format(round(xgb*100),2))
    col2.metric(label="Nombre de clients se désengageant", value=45)

#Essaie de faire deux nouvelles colonnes
#col1, col2 = st.columns(2)
#with col1:
if uploaded_file:
    st.subheader("Causes")
    #3th bloc
    #Chart --> graph feature_importances_
    indices = np.argsort(GBC1.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X.columns[indices][:40],x = GBC1.feature_importances_[indices][:40] , orient='h')
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
