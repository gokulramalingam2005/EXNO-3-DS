## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/bc6129e0-b308-472c-a1f4-969409562c51)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/973d5a60-32dd-4f34-9f9e-68783a458ce4)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/b439f19f-d939-4982-9bc7-42e17b1ca69a)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/3c7e7c85-48f0-4ae4-a914-07fe962b0b02)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/user-attachments/assets/00c0ddc1-96f8-4da2-9d74-236e3348e7b2)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/09b3c25b-51d0-4ac1-be38-d09db2da0083)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/05238fa9-acee-45b5-a17f-c4c1bd3ef5ba)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/61a1cf9a-ffb8-4f1b-8edd-1bc13b5c9ba0)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/user-attachments/assets/5f9667c1-a6f5-4254-b0b6-d535615958f9)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/c341af39-dad1-4e13-9182-c37a86dff91c)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/5c454912-e3f0-4d2b-9b8c-cfb561247efb)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/3c4a66c0-0a3a-4abb-9f84-f76f360296e2)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/b7aa25da-48ed-4074-836b-f5eeed09c7d3)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/eb2e03cc-73dd-4968-a9e5-aa8faff0a3ff)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2a4a91a9-1d1d-4704-ab2f-9bab4800cdd2)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c3e24e02-198e-4dd2-afea-ca5d4772f110)
```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/6828ff62-d637-481a-97b3-d195059619b5)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/5f7d5d8a-8c92-4588-b1a6-e47348b12cb6)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/78879acc-99dc-45b7-863c-d8b150eb3857)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/eb2885b0-cfc9-46da-b510-27513cda9128)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/15a7f058-bf0a-4dd2-9971-7eb4eb8e5588)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c7b49d6d-202e-4c06-bab2-b311d96515f7)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c6e848d0-8682-4f5c-a709-2eaac4e4b3b6)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b063737d-d78d-448c-a2e3-3056df2720a4)

****RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
