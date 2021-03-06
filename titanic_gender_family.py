import pandas as pd
from pandas.api.types import CategoricalDtype
import re

#%% Import data
dataTrain = pd.read_csv('train.csv').rename(columns=lambda x: x.strip().lower())
dataTest = pd.read_csv('test.csv').rename(columns=lambda x: x.strip().lower())
dataBoth = [dataTrain, dataTest]

#%% Treating missing data
for data in dataBoth:
    # Fill missing fare with 0
    data['fare'].fillna(0, inplace=True)
    # Fill missing cabin by other value and keep only cabin letter
    data['cabin'].fillna('Z', inplace=True)
    data.loc[:, 'cabin'] = data['cabin'].apply(lambda x: x[0])
    # Fill missing embakred by most common
    data['embarked'].fillna('S', inplace=True)

#%% Feature engineering

for data in dataBoth:
    # Create title from name and categorize into Mr, Mrs and Master
    data.loc[:, 'title'] = data['name'].apply(lambda x: re.search(r'\s(\w*)\.', x)[1])
    data['title'] = data['title'].replace(['Capt', 'Col', 'Major', 'Don', 'Jonkheer', 'Dr', 'Rev', 'Sir'], 'Mr')
    data['title'] = data['title'].replace(['Mme', 'Countess', 'Dona', 'Lady', 'Mlle', 'Ms', 'Miss'], 'Mrs')
    data['title'] = data['title'].astype(CategoricalDtype(categories=['Master', 'Mr', 'Mrs']))
    # Create lastname from name
    data.loc[:, 'lastname'] = data['name'].apply(lambda x: x.split(sep=',')[0])

# Converting sex to binary
genders = {"male": 0, "female": 1}
for data in dataBoth:
    data['sex'] = data['sex'].map(genders)
    
# Dropping features
for data in dataBoth:
    data.drop(['name', 'age', 'fare', 'sibsp', 'parch', 'cabin', 'embarked'], axis=1, inplace=True)

#%% Survival based on gender and family
dataAll = pd.concat([dataTrain, dataTest], sort=False)
dataAll.reset_index(drop=True, inplace=True)    

# By default assume all females lived and everyone else died
dataAll.loc[dataAll.title!='Mrs', 'prediction'] = False
dataAll.loc[dataAll.title=='Mrs', 'prediction'] = True

# For each family, grouped by lastname
# Females die if all other females and masters in her family die
# Master lives if all other females and masters in his family live
dataAll_grouped_ln = dataAll.loc[dataAll.title!='Mr'].groupby('lastname')
for tnum, df in dataAll_grouped_ln:
    if ((len(df)!=1) & (df.survived.isnull().any())):
        for ind, row in df.iterrows():
            if row.title=='Mrs':
                dataAll.loc[dataAll.passengerid==row.passengerid,
                            'prediction'] = pd.Series([df.drop(ind)['survived'].any(),
                                                       df.drop(ind)['survived'].all()]).any()
            else:
                dataAll.loc[dataAll.passengerid==row.passengerid,
                            'prediction'] = pd.Series([df.drop(ind)['survived'].any(),
                                                       df.drop(ind)['survived'].all()]).all()

# Same as above but grouped by ticket number                                                       
dataAll_grouped_ticket = dataAll.loc[dataAll.title!='Mr'].groupby('ticket')
for tnum, df in dataAll_grouped_ticket:
    if ((len(df)!=1) & (df.survived.isnull().any())):
        for ind, row in df.iterrows():
            if row.title=='Mrs':
                dataAll.loc[dataAll.passengerid==row.passengerid,
                            'prediction'] = pd.Series([df.drop(ind)['survived'].any(),
                                                       df.drop(ind)['survived'].all()]).any()
            else:
                dataAll.loc[dataAll.passengerid==row.passengerid,
                            'prediction'] = pd.Series([df.drop(ind)['survived'].any(),
                                                       df.drop(ind)['survived'].all()]).all()

dataAll.loc[:, 'prediction'] = dataAll['prediction'].astype(int)

#%% Separating train and test set
impTrain = dataAll.loc[pd.notna(dataAll['survived'])].copy()
impTrain.loc[:, 'survived'] = impTrain['survived'].astype(int)
impTest = dataAll.loc[pd.isna(dataAll['survived'])].copy().reset_index(drop=True)

#%% Saving prediction file
dataPred = pd.DataFrame()
dataPred['PassengerId'] = dataTest['passengerid'].copy()
dataPred['Survived'] = impTest.prediction
dataPred.to_csv('submission.csv', index=False)
