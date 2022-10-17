
import streamlit as st
import pandas as pd
import numpy as np
import pickle



st.header("Palmer Penguins Prediction App")

# Read penguins CSV file

df1=pd.read_csv(r'https://github.com/foxtrot20/Projects_Convergytics/blob/master/penguins-heroku/penguins_cleaned.csv')
df1.head()


# Taking user input for independent variables

st.sidebar.header("Choose independent variables")
island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
bill_length_mm = st.sidebar.slider('Bill length (mm)', df1['culmen_length_mm'].min(),df1['culmen_length_mm'].max(),43.9)
bill_depth_mm = st.sidebar.slider('Bill depth (mm)', df1['culmen_depth_mm'].min(),df1['culmen_depth_mm'].max(),17.2)
flipper_length_mm = st.sidebar.slider('Flipper length (mm)', df1['flipper_length_mm'].min(),df1['flipper_length_mm'].max(),201.0)
body_mass_g = st.sidebar.slider('Body mass (g)', df1['body_mass_g'].min(),df1['body_mass_g'].max(),4207.0)
sex = st.sidebar.selectbox('Sex',('MALE','FEMALE'))

User_Input_dict={'island':island,'culmen_length_mm':bill_length_mm,'culmen_depth_mm':bill_depth_mm,'flipper_length_mm':flipper_length_mm,'body_mass_g':body_mass_g,'sex':sex}

User_Input_df=pd.DataFrame(User_Input_dict,index=[0])


# Dropping species column
df1.drop(columns=['species'],inplace=True)

# Concatinating df1 with User_Input_df
df2=pd.concat([User_Input_df,df1],axis=0,ignore_index=True)

# Encoding categorical variables in df2
df3=pd.get_dummies(df2,columns=['island','sex'],drop_first=False)
df4 = df3[:1]
#df4.head()

# Loading pickle object classifier
load_obj = pickle.load(open('penguins_obj.pkl', 'rb'))

# Making prediction
prediction=load_obj.predict(df4)

# Prediction probability
probability=load_obj.predict_proba(df4)



st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(probability)


                                        
                                                                                                                                                           