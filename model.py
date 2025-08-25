import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def load_dataset():
    df = pd.read_csv('flight_dataset.csv')
    return df

def check_dataset():
    df = load_dataset()
    df.info()
    df.isna().sum()

def isDuplicated():
    df = load_dataset()
    df.duplicated()

def model():
    df = load_dataset()
    label_encoder = LabelEncoder()
    df[['Airline', 'Source', 'Destination']] = df[['Airline', 'Source', 'Destination']].apply(label_encoder.fit_transform)
    filtered_df = outlier_removal(df)

    x = filtered_df.drop(columns=['Price'])
    y = filtered_df['Price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    

def outlier_removal(df):
    Q1_price = df['Price'].quantile(0.25)
    Q3_price = df['Price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price

    Q1_duration = df['Duration_hours'].quantile(0.25)
    Q3_duration = df['Duration_hours'].quantile(0.75)
    IQR_duration = Q3_duration - Q1_duration

    lower_bound_price = Q1_price - 1.5 * IQR_price
    upper_bound_price = Q3_price + 1.5 * IQR_price

    lower_bound_duration = Q1_duration - 1.5 * IQR_duration
    upper_bound_duration = Q3_duration + 1.5 * IQR_duration


    filtered_df = df[
    (df['Price'] >= lower_bound_price) & (df['Price'] <= upper_bound_price) &
    (df['Duration_hours'] >= lower_bound_duration) & (df['Duration_hours'] <= upper_bound_duration)
    ]

    return filtered_df

model()