import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("House Price Guesser - Works in Any State")
st.write("Pick a state, type in a house, get a price.")

url = "https://raw.githubusercontent.com/MayBlaze/Usa_housing/main/USA_Housing.csv"
df = pd.read_csv(url)
df["State"] = df["Address"].str.extract(r", ([A-Z]{2}) ")
df = pd.get_dummies(df, columns=["State"])

X = df.drop(columns=["Price", "Address"])
y = df["Price"]

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X, y)

st.write("Model trained on 5000 houses across America. MAE about $96k.")

state_cols = [c for c in X.columns if c.startswith("State_")]
states = sorted([c.replace("State_", "") for c in state_cols])
state = st.selectbox("Pick State", states)

income = st.number_input("Avg Area Income", value=70000)
house_age = st.number_input("Avg Area House Age", value=6.0)
rooms = st.number_input("Avg Area Number of Rooms", value=7.0)
bedrooms = st.number_input("Avg Area Number of Bedrooms", value=4.0)
population = st.number_input("Area Population", value=30000)

if st.button("Get Price"):
    new_row = pd.DataFrame(0, index=[0], columns=X.columns)
    new_row["Avg. Area Income"] = income
    new_row["Avg. Area House Age"] = house_age
    new_row["Avg. Area Number of Rooms"] = rooms
    new_row["Avg. Area Number of Bedrooms"] = bedrooms
    new_row["Area Population"] = population
    new_row[f"State_{state}"] = 1
    pred = forest_model.predict(new_row)[0]
    st.success(f"Estimated Price: ${pred:,.0f}")
