
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[193]:


os.chdir('C:/Users/sarath chandra/Desktop/data science/project 2')


# In[194]:


data=pd.read_excel('project.xls')


# In[195]:


val=data.columns
print(val)


# In[196]:


missing_val = pd.DataFrame(data.isnull().sum())


# In[198]:


for i in val:
    data[i]=data[i].fillna(data[i].mean())


# In[201]:


f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = data.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[202]:


catg=['Reason for absence','Day of the week','Seasons','Month of absence',
     'Disciplinary failure','Education','Son','Social drinker','Social smoker','Absenteeism time in hours']
cont=['ID','Transportation expense','Distance from Residence to Work','Service time','Age',
      'Work load Average','Hit target','Pet','Weight','Height','Body mass index']


# In[203]:


cname=data.columns


# In[206]:


for i in val:
    data[i]=data[i].fillna(data[i].mean())


# In[207]:


for i in catg:
    data[i]=data[i].astype('category')
catg.remove('Absenteeism time in hours')


# In[208]:



plt.boxplot(data['Height'])


# In[209]:


a=[]
#print(chi2_contingency(pd.crosstab(data.loc[:,'Reason for absence'],data[:,'Absenteeism time in hours'])))
for i in catg:
    chi2, p, dof, ex=chi2_contingency(pd.crosstab(data.loc[:,i],data.loc[:,'Absenteeism time in hours']))
    a.append(p)
    print(i,"categories",p)


# In[210]:


q=0
for i in catg:
    if(a[q]<0.05):
        print(a[q])
        print(i)
    q=q+1


# In[212]:


data=data.drop(['Social smoker','Education'],axis=1)
for i in cont:
    print(i)
    data[i] = (data[i] - min(data[i]))/(max(data[i]) - min(data[i]))


# In[213]:


Q.index.shape


# In[214]:
for i in cname:
    q75, q25 = np.percentile(data[i], [75 ,25])

# #Calculate IQR
    iqr = q75 - q25

 #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)

# #Replace with NA
    data.loc[data[i] < minimum,:i] = np.nan
    data.loc[data[i] > maximum,:i] = np.nan


# In[215]:


name=data.loc[data['Absenteeism time in hours']==0,:]
msk = np.random.rand(len(name)) < 0.8
train = name[msk]
test = name[~msk]
X_train=train.iloc[:,0:17]
Y_train=train.iloc[:,18]
Y_train=Y_train.astype('int')
X_test=test.iloc[:,0:17]
Y_test=test.iloc[:,18]
Y_test=Y_test.astype('int')
print(X_train.shape)
for i in Q.index:
    name1=data.loc[data['Absenteeism time in hours']==i,:]
    #print(i)
    msk = np.random.rand(len(name1)) < 0.8
    train = name1[msk]
    test = name1[~msk]
    X_train1=train.iloc[:,0:17]
    print(X_train1.shape)
    Y_train1=train.iloc[:,18]
    Y_train1=Y_train1.astype('int')
    X_test1=test.iloc[:,0:17]
    Y_test1=test.iloc[:,18]
    Y_test1=Y_test1.astype('int')
    X_train=pd.concat([X_train,X_train1])
    print("final")
    print(X_train.shape)
    Y_train=pd.concat([Y_train,Y_train1])
    X_test=pd.concat([X_test,X_test1])   
    Y_test=pd.concat([Y_test,Y_test1]) 


# In[140]:


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.5)


# In[216]:


C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,Y_train)

#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[217]:


cm=pd.crosstab(Y_test,C50_Predictions)


# In[218]:


cm


# In[22]:


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_train)
roc_auc = metrics.auc(fpr, tpr)


# In[221]:



cm.columns
cm.index


# In[42]:


tp=0
tot=0
for i in cm.columns:
    for j in cm.columns:
        tot=tot+cm[i][j]
        if i==j:
            tp=tp+cm[i][j]
print(tp/tot)


# In[222]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, Y_train)


# In[223]:


RF_Predictions = RF_model.predict(X_test)


# In[224]:


CM = pd.crosstab(Y_test, RF_Predictions)


# In[225]:

for i in cm.columns:
    for j in cm.columns:
        tot=tot+cm[i][j]
        if i==j:
            tp=tp+cm[i][j]
print(tp/tot)



# In[226]:


from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, Y_train)


# In[227]:


NB_Predictions = NB_model.predict(X_test)


# In[228]:


CM = pd.crosstab(Y_test, NB_Predictions)


# In[229]:

for i in cm.columns:
    for j in cm.columns:
        tot=tot+cm[i][j]
        if i==j:
            tp=tp+cm[i][j]
print(tp/tot)



# In[230]:


from sklearn.neighbors import KNeighborsClassifier


# In[231]:


KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(X_train, Y_train)


# In[232]:


KNN_Predictions = KNN_model.predict(X_test)


# In[233]:


CM = pd.crosstab(Y_test, KNN_Predictions)


# In[234]:


CM

