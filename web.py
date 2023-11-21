import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing

model = pickle.load(open('./model/cbr.pkl','rb'))

teams = ['Australia',
         'India',
         'Bangladesh',
         'New Zealand',
         'South Africa',
         'England',
         'West Indies',
         'Afghanistan',
         'Pakistan',
         'Sri Lanka']

cities = ['Colombo',
'Mirpur',
'Johannesburg',
'Dubai',
'Auckland',
'Cape Town',
'London',
'Pallekele',
'Barbados', 
'Sydney',
'Melbourne',
'Durban',
'St Lucia',
'Wellington',
'Lauderhill',
'Hamilton',
'Centurion',
'Manchester',
'Abu Dhabi',
'Mumbai',
'Nottingham',
'Southampton',
'Mount Maunganui',
'Chittagong',
'Kolkata',
'Lahore',
'Delhi',
'Nagpur',
'Chandigarh',
'Adelaide',
'Bangalore',
'St Kitts',
'Cardiff',
'Christchurch',
'Trinidad']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling Team',sorted(teams))

city = st.selectbox('Select City',sorted(cities))
col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for overs>5)')
with col5:
    wickets = st.number_input('Wickets Out')

last_five = st.number_input('Runs scored in last five overs')

if st.button('predict score'):
    balls_left = 120-(overs*6)
    wickets_left = 10-wickets
    crr = current_score/overs

    final_df1 = pd.DataFrame(
        {'batting_team':[batting_team],
         'bowling_team':[bowling_team],
         'city':city,
         'current_score':[current_score],
         'balls_left':[balls_left],
         'wickets_left':[wickets],
         'crr':[crr],
         'last_five':[last_five],
         })
    # Converting type of columns to category 
    final_df1['batting_team'] = final_df1['batting_team'].astype('category') 
    final_df1['bowling_team'] = final_df1['bowling_team'].astype('category') 
    final_df1['city'] = final_df1['city'].astype('category') 
  
    # Assigning numerical values and storing it in another columns 
    final_df1['batting_team'] = final_df1['batting_team'].cat.codes 
    final_df1['bowling_team'] = final_df1['bowling_team'].cat.codes
    final_df1['city'] = final_df1['city'].cat.codes
    
    # Create an instance of One-hot-encoder 
    enc = OneHotEncoder() 
  
    # Passing encoded columns 
    enc_data = pd.DataFrame(enc.fit_transform( 
        final_df1[['batting_team', 'bowling_team','city']]).toarray()) 
  
    # Merge with main 
    New_df = final_df1.join(enc_data) 
    final_df2= pd.DataFrame(New_df)
    final_df2 = final_df2[['batting_team','bowling_team',
                           'city','current_score','balls_left','wickets_left','crr','last_five']]
    
    result = model.predict(final_df2)
    result = preprocessing.normailize(result)
    st.header("Predicted Score - ", str(int(result[0])))


