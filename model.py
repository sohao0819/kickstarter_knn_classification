import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class KickstarterModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        # convert to usd
        df['goal_usd'] = df['goal'] * df['static_usd_rate']

        # remove outlier
        Q1 = np.percentile(df['goal_usd'], 25)
        Q3 = np.percentile(df['goal_usd'], 75)
        IQR = Q3 - Q1
        floor = Q1 - 1.5 * IQR
        ceiling = Q3 + 1.5 * IQR
        data_filtered = df[~((df['goal_usd'] < floor) |
                             (df['goal_usd'] > ceiling))]

        # category json
        data_filtered['slug_from_category'] = (data_filtered.apply(
            lambda x: json.loads(x.category)['slug'], axis=1))
        data_filtered['project_category'] = (data_filtered['slug_from_category'].
                                             str.split(pat='/').str[0])
        data_filtered['sub_category'] = (data_filtered['slug_from_category'].
                                         str.split(pat='/').str[1])

        # duration
        data_filtered['deadline'] = pd.to_datetime(
            data_filtered['deadline'], unit='s')
        data_filtered['launched_at'] = pd.to_datetime(
            data_filtered['launched_at'], unit='s')
        data_filtered['duration'] = abs(
            data_filtered['deadline']-data_filtered['launched_at']).dt.days

        # country,project_category,sub_category encode
        le = preprocessing.LabelEncoder()
        data_filtered['country_encoded'] = (le.fit_transform(
                                            data_filtered['country']))
        data_filtered['project_category_encoded'] = (le.fit_transform(
                                                     data_filtered['project_category']))
        data_filtered['sub_category_encoded'] = (le.fit_transform(
                                                 data_filtered['sub_category']))

        # data normalisation
        scaler = StandardScaler()
        # Apply auto-scaling (or any other type of scaling) and cast to DataFrame
        data_filtered_pre_scale = data_filtered[['goal_usd', 'duration']]
        data_filtered_after_scale = pd.DataFrame(
            scaler.fit_transform(data_filtered_pre_scale),
            columns=data_filtered_pre_scale.columns,
            index=data_filtered_pre_scale.index)

        # create X
        X_train = pd.concat(
            [data_filtered_after_scale,
             data_filtered[['country_encoded',
                            'project_category_encoded', 'sub_category_encoded']]], axis=1)

        # create Y
        y_train = (data_filtered["state"].apply(
            lambda x: 1 if x == "successful" else 0))

        return X_train, y_train

    def fit(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=36)
        self.model.fit(X, y)

    def preprocess_unseen_data(self, df):

        # convert to usd
        df['goal_usd'] = df['goal'] * df['static_usd_rate']

        # category json
        df['slug_from_category'] = df.apply(
            lambda x: json.loads(x.category)['slug'], axis=1)
        df['project_category'] = (df['slug_from_category'].
                                  str.split(pat='/').str[0])
        df['sub_category'] = (df['slug_from_category']
                              .str.split(pat='/').str[1])

        # duration
        df['deadline'] = pd.to_datetime(df['deadline'], unit='s')
        df['launched_at'] = pd.to_datetime(df['launched_at'], unit='s')
        df['duration'] = abs(df['deadline']-df['launched_at']).dt.days

        # country,project_category,sub_category encode
        le = preprocessing.LabelEncoder()
        df['country_encoded'] = le.fit_transform(df['country'])
        df['project_category_encoded'] = le.fit_transform(
            df['project_category'])
        df['sub_category_encoded'] = le.fit_transform(df['sub_category'])

        # data normalisation
        scaler = StandardScaler()
        # Apply auto-scaling (or any other type of scaling) and cast to DataFrame
        data_filtered_pre_scale = df[['goal_usd', 'duration']]
        data_filtered_after_scale = pd.DataFrame(
            scaler.fit_transform(data_filtered_pre_scale),
            columns=data_filtered_pre_scale.columns,
            index=data_filtered_pre_scale.index)

        # create X
        X_test = pd.concat(
            [data_filtered_after_scale,
             df[['country_encoded', 'project_category_encoded',
                 'sub_category_encoded']]], axis=1)

        return X_test

    def predict(self, X):

        return self.model.predict(X)
