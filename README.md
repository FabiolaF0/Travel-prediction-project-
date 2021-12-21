# Travel-prediction-project-
Final project in team where we want to predict if a client will churn or not within an operator tour (batch014).

                            ------------------------------------------------------------------------------------------
![image (1)](https://user-images.githubusercontent.com/92126149/146995706-5f2f0606-790c-40d6-a037-e6e8310dcf9c.png)

# Tour opérateur

**Contexte**: 

  Bien avant la crise de la Covid, de nombreux agents économiques passaient par des tours opérateurs afin de préparer leur voyage. Leurs actions passaient par la   réservation d'un vol simple à la réservation de plus de services comme par exemple la réservation d'un hôtel. Selon l'expérience client, plus l'expérience avec   un tour opérateur quelconque a été bonne plus il tendra à réitérer l'expérience en réservant à nouveau et à davantage se fidéliser. 
  À contrario si l'expérience client a été mauvaise, le client ne renouvelera pas sa réservation voir il peut annuler à un mois de sa réservation ce qui 
  consistera à une perte d'un ou plusieurs clients.

**Problématique** : 

  Comment anticiper les comportements des clients sur un tour opérateur afin de les fidéliser (et de renforcer leur fidélité) dans le but d'éviter la perte de       clients ? Churn

**Pourquoi ce sujet ?**

  Durant la crise sanitaire les frontières dans le monde entier ont été fermées conduisant à l'arrêt des voyages et de facto à la suspension des réservations des   voyages. Certains tours opérateurs n'ont pas réussi à tenir. De plus, selon les solutions mises en place durant cette crise, certains groupes de clients sont     restés alors que d'autres ce sont défidéliser. D'ores et déjà, nous pouvons avoir des idées de certains comportements des consommateurs durant une crise.

  L'idée ici est de prédire si un client hors temps de crise sanitaire restera fidèle ou non à un tour opérateur.

**Quelles techniques en vue ?** Utiliser un modèle de prédiction assez robuste. Jouer sur le clustering. 

**Détails du dataset**:

- Age = Age of user 
- Frequent Flyer = Whether Customer takes frequent flights
- Annual Income Class = Class of annual income of user
- Service Opted = Number of times services opted during recent years
- Account Synced to Social Media = Whether Company Account Of User Synchronised to Their Social Media
- Booked Hotel or not = Whether the customer book lodgings/Hotels using company services
- Target = 1- Customer Churns 
           0- Customer Doesnt Churn 

Voici le lien de notre application déployée : https://share.streamlit.io/fabiolaf0/travel-prediction-project-/main/app_final.py
