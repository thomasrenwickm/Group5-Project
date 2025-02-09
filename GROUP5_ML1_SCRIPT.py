# Importing the necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.write("""
# Predicting a Penguin's Body Mass
""")
st.write('---')

# Reading the dataset
penguins_df = pd.read_csv('penguins_cleaned.csv')

# Feature engineering
df_nums = penguins_df.select_dtypes(exclude='object')
df_objs = penguins_df.select_dtypes(include='object')

# Converting categorical columns to numerical (One-Hot Encoding)
df_objs = pd.get_dummies(df_objs, drop_first=True)

# Merging numeric and categorical features
final_df = pd.concat([df_nums, df_objs], axis=1)

# Splitting labels and features
y = final_df['body_mass_g']
X = final_df.drop('body_mass_g', axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Streamlit function to collect user input
def user_input_features():
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', float(X['bill_length_mm'].min()), float(X['bill_length_mm'].max()), float(X['bill_length_mm'].mean()))
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', float(X['bill_depth_mm'].min()), float(X['bill_depth_mm'].max()), float(X['bill_depth_mm'].mean()))
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', float(X['flipper_length_mm'].min()), float(X['flipper_length_mm'].max()), float(X['flipper_length_mm'].mean()))

    # Categorical feature selection using radio buttons
    species = st.sidebar.radio('Species', ['Adelie', 'Chinstrap', 'Gentoo'])
    island = st.sidebar.radio('Island', ['Biscoe', 'Dream', 'Torgersen'])
    sex = st.sidebar.radio('Sex', ['Female', 'Male'])

    # Encoding categorical variables based on dummy encoding
    species_Chinstrap = 1 if species == 'Chinstrap' else 0
    species_Gentoo = 1 if species == 'Gentoo' else 0
    island_Dream = 1 if island == 'Dream' else 0
    island_Torgersen = 1 if island == 'Torgersen' else 0
    sex_male = 1 if sex == 'Male' else 0  # Female is the reference category (0)

    # Creating the user input dataframe
    data = {
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'species_Chinstrap': species_Chinstrap,
        'species_Gentoo': species_Gentoo,
        'island_Dream': island_Dream,
        'island_Torgersen': island_Torgersen,
        'sex_male': sex_male
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Collect user input
df = user_input_features()
st.header('Specified Input Parameters')
st.write(df)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Scale user input
df_scaled = scaler.transform(df.values.reshape(1, -1))

# Predict using the model
prediction = model.predict(df_scaled)

# Display prediction
st.header('Prediction of Body Mass (Grams)')
st.write(f"Predicted Body Mass: **{prediction[0]:.2f} grams**")
st.write('---')

# Evaluate the model
y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

st.write(f"**Mean Absolute Error (MAE):** {MAE:.2f}")
st.write(f"**Mean Squared Error (MSE):** {MSE:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {RMSE:.2f}")
