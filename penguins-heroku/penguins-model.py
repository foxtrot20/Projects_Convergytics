import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


# Reading the cleaned csv into a dataframe
# Change url to local csv when running locally
df1=pd.read_csv('https://raw.githubusercontent.com/foxtrot20/Projects_Convergytics/master/penguins-heroku/penguins_cleaned.csv')
df1.head()

# Mapping the target

Target_dict={'Adelie':0,'Chinstrap':1,'Gentoo':2}

# Creating function for mapping the target

def Target_Mapper(species):
    return Target_dict[species]


# Encoding the species column
df1['species'] = df1['species'].apply(Target_Mapper)


# One hot encoding the island and sex columns
df1=pd.get_dummies(df1,columns=['island','sex'],drop_first=False)
df1.head()


# Seperating the dependent and independent variables
Y=df1['species']
X=df1.drop(columns=['species'])


# Creating a random forest model object and fitting it over X and Y
obj=RandomForestClassifier()

obj.fit(X,Y)


# Dumping the model object into a pickle file

pickle.dump(obj, open('penguins_obj.pkl', 'wb'))

