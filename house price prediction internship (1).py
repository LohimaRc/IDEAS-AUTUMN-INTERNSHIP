#!/usr/bin/env python
# coding: utf-8

# # IDEAS- Institute of Data Engineering ,Analytics and Science Foundation
# project submitted by - Lohima Roy Choudhury

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_csv(r"C:\Users\User\Downloads\house_price_india (1).csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


house_data_missing= df.copy()
## each column missing values are inserted, 20% sample of each column is been drawn and its index is noted to replace those rows with nan
for i in house_data_missing.columns:
 house_data_missing[i].loc[house_data_missing[i].sample(frac=0.2).index]= np.nan
house_data_missing


# In[9]:


#Q1. Try inserting a missing value to a specific column of your choice?
import numpy as np

house_data_missing = df.copy()

# randomly pick 20% indices from the column
missing_indices = house_data_missing["Postal Code"].sample(frac=0.2, random_state=42).index  

# replace those with NaN
house_data_missing.loc[missing_indices, "Postal Code"] = np.nan

house_data_missing.head()


# In[10]:


house_data_missing.info()


# In[11]:


## Date column wrong input in the data so we drop it and keep inside the actual data
house_data_missing.drop(['Date','Longitude','Renovation Year','Postal Code', 'Lattitude','living_area_renov', 'lot_area_renov'],axis=1,inplace=True)


# In[12]:


house_data_missing.columns


# In[13]:


#Q2. Show statistics about the data of only numeric columns**
house_data_missing.describe()


# In[14]:


plt.plot(house_data_missing.index,house_data_missing['Price'],marker='o',linestyle='-',color='b', label="Distribution of Price for all houses")
plt.show()


# **Q3. Find the distribution of area (total area) of houses (use Seaborn distplot)**

# In[15]:


df['total_area'] = df['living area'] + df['Area of the basement']

# plot distribution
plt.figure(figsize=(10,6))
sns.distplot(df['total_area'], bins=50, kde=True, color='blue')
plt.title("Distribution of Total House Area", fontsize=16)
plt.xlabel("Total Area (sq ft)")
plt.ylabel("Density")
plt.show()


# In[16]:


#***Checking for duplicate rows***
house_data_missing.duplicated().sum()


# In[17]:


#***Checking missing values***
house_data_missing.isna().sum()


# In[18]:


#***Technique 1: Remove missing value rows***
house_data_missing1= house_data_missing.dropna()
house_data_missing1.isna().sum()


# **Q4. Try replacing the missing values with the standard deviation of each column**

# In[19]:


house_data_missing2= house_data_missing.copy()
for cols in house_data_missing2.columns:
  house_data_missing2[cols]= house_data_missing2[cols].fillna(np.mean(house_data_missing2[cols]))
house_data_missing2.isna().sum()


# In[20]:


house_data_missing2= house_data_missing.copy()
for cols in house_data_missing2.columns:
  house_data_missing2[cols]= house_data_missing2[cols].fillna(np.std(house_data_missing2[cols]))
house_data_missing2.isna().sum()


# ***Technique 3: Interpolation***

# In[19]:


house_data_missing3= house_data_missing.interpolate(method='linear')
house_data_missing2.isna().sum()


# In[20]:


#Q5. Try replacing the missing values using interpolation with the polynomial method**

house_data_missing3= house_data_missing.interpolate(method='polynomial',order =2)
house_data_missing3.isna().sum()


# # Technique 4: KNN imputation***
# 

# In[21]:


from sklearn.impute import KNNImputer
imputed_vals= KNNImputer(n_neighbors=5)
imputed_data= pd.DataFrame(imputed_vals.fit_transform(house_data_missing),columns= house_data_missing.columns)
imputed_data.isna().sum()


# **Q6. Perform replacing missing values with KNN imputers on scaled data. Also, inverse the scaled data to get the original data.**
# 

# In[22]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaled_data = scaler.fit_transform(house_data_missing)
imputer = KNNImputer(n_neighbors=5)  
imputed_scaled = imputer.fit_transform(scaled_data)
imputed_data = scaler.inverse_transform(imputed_scaled)

# Step 4: Convert back to DataFrame
house_data_knn = pd.DataFrame(imputed_data, columns=house_data_missing.columns)

# Check if any missing values remain
print(house_data_knn.isna().sum())
house_data_knn.head()
scaled_df = pd.DataFrame(scaled_data,columns=house_data_missing.columns)


# Changing datatype

# In[25]:


house_data_missing['Number of schools nearby']=house_data_missing['Number of schools nearby'].astype('int')


# In[24]:


import seaborn as sns

imputed_data = pd.DataFrame(imputed_data, columns=house_data_missing.columns)

plt.figure(figsize=(15,5))
sns.heatmap(imputed_data.corr(), annot=True, fmt='.2f', cmap='viridis', linewidths=0.5,
            cbar_kws={"shrink": 0.88})
plt.show()


# Q8. Find features that are highly correlated with the area of the house (excluding the basement)

# Area of the house positively correlated with number of bathrooms,living area,grade of the house,price

# In[25]:


sns.pairplot(imputed_data)
plt.show()


# In[29]:


imputed_data.groupby('Number of schools nearby')['Price'].agg(np.mean)
## OR
np.mean(imputed_data[imputed_data['Number of schools nearby']==3]['Price'])


# **Q10. Find the average area of houses having 5 bedrooms**

# In[32]:


imputed_data.groupby('number of bedrooms')['living area'].agg(np.mean)
## OR
np.mean(imputed_data[imputed_data['number of bedrooms']==5]['living area'])


# **Selecting the features for predicting

# In[27]:


features=[]
for cols in imputed_data.iloc[:,:-1].columns:
  if (imputed_data['Price'].corr(imputed_data[cols]))>0.50:
    features.append(cols)
print(features)


# In[28]:


X= imputed_data[features]
y= imputed_data.iloc[:,-1]


# In[ ]:


**Q11. Show the pairwise distribution of X and y**


# In[35]:


df = X.copy()
df['y'] = y

sns.pairplot(df, diag_kind="kde", plot_kws={'alpha':0.6})
plt.suptitle("Pairwise distribution of X and y", y=1.02, fontsize=16)
plt.show()


# ***Splitting datasets into training and testing***

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=123)


# **Q12. Split the data as 60% training and 40% testing**

# In[31]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.4,random_state=123)


# ***Fitting the linear regression model and predicting***

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score


# In[32]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score


# In[33]:


MSE= mean_squared_error(y_test,y_pred)
R_square= r2_score(y_test,y_pred)


# In[34]:


print(MSE)
print("==============")
print(R_square)


# **Q13. Try model building and predicting with some other dataset of your choice**

# In[2]:


#I took another dataset "Food Delivery Time Prediction" . The objective was to predict food delivery times based on customer location, restaurant location, weather, traffic, and other factors.
df1= pd.read_csv(r"C:\Users\User\Downloads\Food_Delivery_Time_Prediction (1).csv")
df1.head()


# In[3]:


df1.info()


# In[4]:


df1.describe()


# In[5]:


print(df1.isnull().sum())


# In[29]:


from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[34]:


from sklearn.preprocessing import LabelEncoder

label_cols = ["Weather_Conditions", "Traffic_Conditions", "Order_Time", "Vehicle_Type","Restaurant_Rating","Order_Cost","Tip_Amount"]
le = LabelEncoder()

for col in label_cols:
    df1[col] = le.fit_transform(df1[col])


# In[31]:


scaler = StandardScaler()
num_cols = ['Distance', 'Order_Cost', 'Delivery_Time']
df1[num_cols] = scaler.fit_transform(df1[num_cols])


# In[36]:


# Correlation matrix
df1_cor = df1[['Delivery_Time',"Distance","Weather_Conditions","Traffic_Conditions","Delivery_Person_Experience","Order_Cost","Order_Time","Vehicle_Type","Tip_Amount"]]
plt.figure(figsize=(15, 5))
sns.heatmap(df1_cor.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[38]:


x = df1[["Distance","Weather_Conditions","Traffic_Conditions","Delivery_Person_Experience","Order_Cost","Vehicle_Type","Tip_Amount"]]
y = df1['Delivery_Time']


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=123)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
MSE= mean_squared_error(y_test,y_pred)
R_square= r2_score(y_test,y_pred)
print(MSE)

print(R_square)


# In[ ]:




