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
       
```python
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/9a445ed3-f79e-46ed-8493-a0138abde135)

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/c5ae2314-6f2b-4d93-92b3-f44d1b74015a)



```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/4ae17d2a-aa22-4340-9faf-8567549250f6)



```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/2249ccf3-4a16-462b-b745-677312c7fd42)



```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/d2714505-ceae-48c6-b428-fc421aaa735d)


```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/b4b4c5b2-9bc8-4f41-8649-096999696847)

```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/e56e11b0-9489-41a5-973c-e32fca8f9840)



```python
pip install --upgrade category_encoders
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/0711d42f-4456-4222-8334-f183bc7c2385)



```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/3d2f8b4c-0ffc-4754-8c1b-ad637c727c9b)



```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/781ddd71-1fc6-499b-9234-b83778405580)


```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/6f1877a4-9ba9-45d6-8df2-38fdc103a0ef)



```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/63cbb12a-e9eb-447e-855a-e56c706bbfa9)



```python
df.skew()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/3d04bbce-76dc-4571-8c8d-5aad234c1766)



```python
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/7247340c-6488-4b75-9deb-0ad3f10e03fd)



```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/71ae0399-a828-406a-93a6-0e36cc31e249)


```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/9b500fd0-9b55-4397-b1e8-364652aca983)


```python
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/d243323b-c97e-4c55-a41f-f76d176e6461)


```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/758eaaba-b780-4fee-8487-d8242a9d6148)


```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/4945b8c6-e27d-4526-9032-0c0aeb9ab576)


```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/52a7553c-c1bd-4489-a0cb-b13a27684c23)



```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/3688ed78-4920-4cd4-9e33-4420fc790b8d)



```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/9ef5152c-d766-48e1-857c-a7dbfde4e648)



```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/fde4b296-88ec-46ad-b6f3-2cf2b64a15f2)


```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/57bae70b-8ee0-4ab1-86bf-733d2597089d)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-3-DS/assets/118610231/3987a28b-3816-41b2-9a9d-6a1cedf8382e)




# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.


       
