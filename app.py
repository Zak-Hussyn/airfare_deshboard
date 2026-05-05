import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For Prediction model
import pickle #Loading Model From Drive
import requests

# Loading the data because Streamlit is for Dashboard and you'll have to do everything separately that you used to did in Jupyter Notebook 
df = pd.read_csv('dataset.csv')

# Loading/Downloading Model From Drive
url = "https://drive.google.com/uc?id=1akg6NKpEM4R2NJ5n7QaTzd2aowfuUA23"
response = requests.get(url)

with open("model_new.pkl", "wb") as f:
    f.write(response.content)

model = joblib.load('model_new.pkl')


# Name your project
st.title("BANGLADESH AIRFARE PREDICTION DASHBOARD")
st.caption("*Built using Machine Learning & Streamlit")
st.subheader("ENTER FLIGHT DETAILS")
col1, col2 = st.columns(2)
with col1:
    input_airline = st.selectbox("Airline", df["Airline"].unique())
    input_stops = st.selectbox("Stopover", df["Stopovers"].unique())
    #input_aircraft = st.selectbox("Aircraft Type", df["Aircraft Type"].unique())
    input_class = st.selectbox("Class", df["Class"].unique())
    input_season = st.selectbox("Seasonality", df["Seasonality"].unique())
    #input_source = st.selectbox("Booking Source", df["Booking Source"].unique())
with col2:
    input_duration = st.number_input("Duration (hrs)", min_value=1.0)
    input_base = st.number_input("Base Fare (BDT)", min_value=0, help="Visit **Google Flights** for updated Base Fare. The link is given below")
    input_tax = st.number_input("Tax & Surcharge (BDT)", value=input_base*0.15, help="Default is 15%. You can manually change")
    input_depart = st.number_input("Days Before Departure", min_value=0)
    #input_ddt = st.selectbox("Departure Date & Time", df["Departure Date & Time"].unique())

from sklearn.preprocessing import LabelEncoder
le_airline = LabelEncoder()
le_airline.fit(df['Airline'])
encoded_stops = le_airline.transform([input_airline])[0]

le_stops = LabelEncoder()
le_stops.fit(df['Stopovers'])
encoded_stops = le_stops.transform([input_stops])[0]

le_class = LabelEncoder()
le_class.fit(df['Class'])
encoded_class = le_class.transform([input_class])[0]

le_season = LabelEncoder()
le_season.fit(df['Seasonality'])
encoded_season = le_season.transform([input_season])[0]

if st.button("Predict Fare"):
    test_input = [[
        encoded_airline,
        input_duration,
        encoded_stops,
        0, #encoded_aircraft,
        encoded_class,
        0, #encoded_source,
        input_base,
        0, #input_tax,
        encoded_season,
        input_depart,
        0, #encoded_ddt,
        0,
        0,
        0,
    ]]
    prediction = model.predict(test_input).astype(int)
    st.markdown(
        "<h4 style='text-align: left; '> Predicted Fare </h4'>",
        unsafe_allow_html=True
    )
    st.success(f" {prediction[0]} (BDT)")
    


st.caption("*Refresh before making another prediction")
st.caption("** **Google Flights** : https://www.google.com/travel/flights")

st.markdown(
    "<hr style='border:1px solid #ddd; '>",
    unsafe_allow_html=True
)

# EDA
st.write("EXPLORATORY DATA ANALYSIS")

# Over-all Metrics
st.markdown("## KEY METRICS")
col3, col4, col5, col6, col6a = st.columns(5)
col3.metric("Typical Fare" , df['Total Fare (BDT)'].median().astype(int))
col4.metric("Average Fare" , df['Total Fare (BDT)'].mean().astype(int))
col5.metric("Minimum Fare", df['Total Fare (BDT)'].min().astype(int))
col6.metric("Maximum Fare", df['Total Fare (BDT)'].max().astype(int))
col6a.metric("Fare Volatility", df['Total Fare (BDT)'].std().astype(int))

df['Duration (hrs)'] = pd.to_timedelta(df['Duration (hrs)'], unit='h')
df['Duration_mins'] = df['Duration (hrs)'].dt.total_seconds()/60

# Categorizing flights according to duration by creating another column "duration_category".
bins = [0, 90, 180, 360, float('inf')]
labels = ['Short (<2hr)', 'Medium (2-3hr)', 'Long (3-6hr)', 'Very Long (>6hr)']
df['duration_category'] = pd.cut(df['Duration_mins'], bins=bins, labels=labels)

st.write(df.groupby('duration_category')['Total Fare (BDT)']
      .agg(['count','mean','median','std','min', 'max',]).astype(int)
      .sort_values(by='mean', ascending=False))
st.markdown("---------")

st.sidebar.header('Filter Options')
selected_class = st.sidebar.selectbox(
    'Select Travel Class',
    df['Class'].unique())

filtered_df = df[
    (df['Class'] == selected_class)
    ]

st.write("SECTION - A")
st.subheader("TRAVEL CLASS ANALYSIS")
st.caption("**Use CLASS Filter for this section")

# Metrics for section A
st.markdown("## KEY METRICS")
col7, col8, col9, col10, col11 = st.columns(5)
col7.metric("Typical Fare" , filtered_df['Total Fare (BDT)'].median().astype(int))
col8.metric("Average Fare" , filtered_df['Total Fare (BDT)'].mean().astype(int))
col9.metric("Minimum Fare", filtered_df['Total Fare (BDT)'].min().astype(int))
col10.metric("Maximum Fare", filtered_df['Total Fare (BDT)'].max().astype(int))
col11.metric("Fare Volitality", filtered_df['Total Fare (BDT)'].std().astype(int))

st.subheader(f"Average Ticket Cost By Season: {selected_class} Class")
avg_fare_by_class = filtered_df.groupby('Seasonality')['Total Fare (BDT)'].median()
fig, ax = plt.subplots()
ax.bar(avg_fare_by_class.index, avg_fare_by_class.values)
ax.set_xlabel("Season")
ax.set_ylabel("Median Fare")
st.pyplot(fig)

col14, col15 = st.columns(2)
with col14:
    st.write('Premium Carriers',filtered_df.groupby('Airline')['Total Fare (BDT)']
      .median()
      .sort_values(ascending=False).head(5).astype(int))
with col15:
    st.write('Budget Friendly Fares', filtered_df.groupby('Airline')['Total Fare (BDT)']
      .median()
      .sort_values(ascending=True).head(5).astype(int))
    
# Line Plot
st.subheader(f"Average Fare Trend: {selected_class} Class")
ddd = filtered_df.groupby('Days Before Departure')['Total Fare (BDT)'].median().reset_index().sort_values(by='Total Fare (BDT)', ascending=False).astype(int)

fig, ax = plt.subplots()
sns.lineplot(x='Days Before Departure', y='Total Fare (BDT)', data = ddd)
#plt.title('Fare Trend')
plt.ylabel('Average Fare')
plt.xlabel('Days Before Departure')
st.pyplot(fig)

st.markdown(
    "<hr style='border:1px solid #ddd; '>",
    unsafe_allow_html=True
)

# Fare & Stopovers
st.write("SECTION - B")
st.subheader("FLIGHT FARES & STOPOVERS")
st.caption("**Use STOP Filter for this section")

selected_stops = st.sidebar.selectbox(
    'Select Stop',
    df['Stopovers'].unique())
filtered_df = df[
    (df['Stopovers'] == selected_stops)
    ]

st.markdown("## KEY METRICS")
col16, col17, col18, col19, = st.columns(4)
col16.metric("Total Flights", filtered_df.groupby('Stopovers')['Total Fare (BDT)'].count().astype(int))
col17.metric("Typical Fare (BDT)", filtered_df.groupby('Stopovers')['Total Fare (BDT)'].median().astype(int))
col18.metric("Maximum Fare (BDT)", filtered_df.groupby('Stopovers')['Total Fare (BDT)'].max().astype(int))
col19.metric("Fare Volitality", filtered_df.groupby('Stopovers')['Total Fare (BDT)'].std().astype(int))


st.subheader(f"Seasonal Price Variation: {selected_stops}")
st.write(filtered_df.groupby('Seasonality')['Total Fare (BDT)'].agg(['median', 'std']).astype(int).sort_values(by='median', ascending=False))

avg_fare_by_stop = filtered_df.groupby('Seasonality')['Total Fare (BDT)'].median()
fig, ax = plt.subplots()
ax.bar(avg_fare_by_stop.index, avg_fare_by_stop.values)
ax.set_title("VISUALIZING AVERAGE TICKET COST ")
ax.set_xlabel("Season")
ax.set_ylabel("Median Fare")
st.pyplot(fig)

st.markdown(
    "<hr style='border:1px solid #ddd; '>",
    unsafe_allow_html=True
)

st.write("SECTION - C")
st.subheader("SEASONALITY TRENDS")
st.caption("**Use Seasonality Filter for this section")

selected_season = st.sidebar.selectbox(
    'Select Season',
    df['Seasonality'].unique())

filtered_df = df[
    (df['Seasonality'] == selected_season)
    ]

col20, col21, col22, col23 = st.columns(4)
col20.metric("No. of Flights", filtered_df['Total Fare (BDT)'].count())
col21.metric("Median Fare", filtered_df['Total Fare (BDT)'].median().astype(int))
col22.metric("Maximum Fare", filtered_df['Total Fare (BDT)'].max().astype(int))
col23.metric("Price Volatility", filtered_df['Total Fare (BDT)'].std().astype(int))

st.error(f"TOP 5 MOST EXPENSIVE AIRLINES FOR THE SELECTED SEASON")
st.write(filtered_df.groupby('Airline')['Total Fare (BDT)'].median().astype(int)
    .sort_values(ascending=False).head(5))

st.subheader(f"Average Fare Trend: {selected_season}")
ddd = filtered_df.groupby('Days Before Departure')['Total Fare (BDT)'].median().reset_index().sort_values(by='Total Fare (BDT)', ascending=False).astype(int)

fig, ax = plt.subplots()
sns.lineplot(x='Days Before Departure', y='Total Fare (BDT)', data = ddd)
#plt.title('Fare Trend')
plt.ylabel('Average Fare')
plt.xlabel('Days Before Departure')
st.pyplot(fig)

pivot_fare = df.pivot_table(index='Class', columns='Seasonality', values='Total Fare (BDT)', aggfunc='median', observed=False)
st.subheader('Cross Analysis of Seasonal Fares By Class')
fig, ax = plt.subplots()
sns.heatmap(pivot_fare, annot=True, fmt='.0f', cmap='coolwarm', linewidths=0.5)
plt.title('FLIGHT CATEGORY VS SEASON')
plt.xlabel('Season')
plt.ylabel('Flight Class')
st.pyplot(fig)
st.caption('*It will not change with the filter')
st.markdown("---------")
