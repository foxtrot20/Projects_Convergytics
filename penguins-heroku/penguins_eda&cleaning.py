import pandas as pd
import seaborn as sb


#  Reading the csv file

# Change url to local csv when running locally
df1=pd.read_csv('https://raw.githubusercontent.com/foxtrot20/Projects_Convergytics/master/penguins-heroku/penguins_size.csv')
df1.head(20)


# Null count in columns

df1.isna().sum()


# Replacing NAN's with mean for numerical columns per species

# Creating species wise series
Adelie_species_series=df1['species']=='Adelie'
Chinstrap_species_series=df1['species']=='Chinstrap'
Gentoo_species_series=df1['species']=='Gentoo'

# culmen_length_mm
Adelie_culmen_length_mm_mean=round(df1['culmen_length_mm'][df1['species']=='Adelie'].mean(),1)
df1.loc[Adelie_species_series,'culmen_length_mm']=df1.loc[Adelie_species_series,'culmen_length_mm'].fillna(Adelie_culmen_length_mm_mean)

Chinstrap_culmen_length_mm_mean=round(df1['culmen_length_mm'][df1['species']=='Chinstrap'].mean(),1)
df1.loc[Chinstrap_species_series,'culmen_length_mm']=df1.loc[Chinstrap_species_series,'culmen_length_mm'].fillna(Chinstrap_culmen_length_mm_mean)

Gentoo_culmen_length_mm_mean=round(df1['culmen_length_mm'][df1['species']=='Gentoo'].mean(),1)
df1.loc[Gentoo_species_series,'culmen_length_mm']=df1.loc[Gentoo_species_series,'culmen_length_mm'].fillna(Gentoo_culmen_length_mm_mean)

# culmen_depth_mm
Adelie_culmen_depth_mm_mean=round(df1['culmen_depth_mm'][df1['species']=='Adelie'].mean(),1)
df1.loc[Adelie_species_series,'culmen_depth_mm']=df1.loc[Adelie_species_series,'culmen_depth_mm'].fillna(Adelie_culmen_depth_mm_mean)

Chinstrap_culmen_depth_mm_mean=round(df1['culmen_depth_mm'][df1['species']=='Chinstrap'].mean(),1)
df1.loc[Chinstrap_species_series,'culmen_depth_mm']=df1.loc[Chinstrap_species_series,'culmen_depth_mm'].fillna(Chinstrap_culmen_depth_mm_mean)

Gentoo_culmen_depth_mm_mean=round(df1['culmen_depth_mm'][df1['species']=='Gentoo'].mean(),1)
df1.loc[Gentoo_species_series,'culmen_depth_mm']=df1.loc[Gentoo_species_series,'culmen_depth_mm'].fillna(Gentoo_culmen_depth_mm_mean)

# flipper_length_mm
Adelie_flipper_length_mm_mean=round(df1['flipper_length_mm'][df1['species']=='Adelie'].mean(),1)
df1.loc[Adelie_species_series,'flipper_length_mm']=df1.loc[Adelie_species_series,'flipper_length_mm'].fillna(Adelie_flipper_length_mm_mean)

Chinstrap_flipper_length_mm_mean=round(df1['flipper_length_mm'][df1['species']=='Chinstrap'].mean(),1)
df1.loc[Chinstrap_species_series,'flipper_length_mm']=df1.loc[Chinstrap_species_series,'flipper_length_mm'].fillna(Chinstrap_flipper_length_mm_mean)

Gentoo_flipper_length_mm_mean=round(df1['flipper_length_mm'][df1['species']=='Gentoo'].mean(),1)
df1.loc[Gentoo_species_series,'flipper_length_mm']=df1.loc[Gentoo_species_series,'flipper_length_mm'].fillna(Gentoo_flipper_length_mm_mean)

# body_mass_g
Adelie_body_mass_g_mean=round(df1['body_mass_g'][df1['species']=='Adelie'].mean(),1)
df1.loc[Adelie_species_series,'body_mass_g']=df1.loc[Adelie_species_series,'body_mass_g'].fillna(Adelie_body_mass_g_mean)

Chinstrap_body_mass_g_mean=round(df1['body_mass_g'][df1['species']=='Chinstrap'].mean(),1)
df1.loc[Chinstrap_species_series,'body_mass_g']=df1.loc[Chinstrap_species_series,'body_mass_g'].fillna(Chinstrap_body_mass_g_mean)

Gentoo_body_mass_g_mean=round(df1['body_mass_g'][df1['species']=='Gentoo'].mean(),1)
df1.loc[Gentoo_species_series,'body_mass_g']=df1.loc[Gentoo_species_series,'body_mass_g'].fillna(Gentoo_body_mass_g_mean)

df1.isna().sum()


# Variable analysis for numerical columns
df1.describe()

# Replacing NAN's with mode for categorical columns per species

# Adelie value counts
print("Adelie value counts")
df1['sex'][df1['species']=='Adelie'].value_counts(dropna=False)

# Chinstrap value counts
print("Chinstrap value counts")
df1['sex'][df1['species']=='Chinstrap'].value_counts(dropna=False)

# Gentoo value counts
print("Gentoo value counts")
df1['sex'][df1['species']=='Gentoo'].value_counts(dropna=False)

# Replacing NAN's with mode for Adelie
df1.loc[Adelie_species_series,'sex']=df1.loc[Adelie_species_series,'sex'].fillna('MALE')

# Replacing NAN's and . with mode for Gentoo
df1.loc[Gentoo_species_series,'sex']=df1.loc[Gentoo_species_series,'sex'].fillna('MALE')

df1.loc[Gentoo_species_series,'sex']=df1.loc[Gentoo_species_series,'sex'].replace('.','MALE')

# Final check for null values
df1.isna().sum()

# One hot encoding
df2=pd.get_dummies(df1,columns=['island','sex'],drop_first=False)
df2.head()

# Correlation Plot

corr=df2.corr()
sb.heatmap(corr,cmap='Blues',annot=True)

# Histograms

sb.histplot(data=df1, x="culmen_length_mm", kde=True)

sb.histplot(data=df1, x="culmen_depth_mm", kde=True)

sb.histplot(data=df1, x="flipper_length_mm", kde=True)

sb.histplot(data=df1, x="body_mass_g", kde=True)

df1.head()

df1.to_csv(r'/home/anvit/Projects_Convergytics/penguins-heroku/penguins_cleaned.csv',index=False)


