
# coding: utf-8

# # Employee Absenteeism

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from graphviz import Source


# In[2]:


# Reading data
emp_data = pd.read_excel("Absenteeism_at_work_Project.xls")
emp_data.head()


# In[3]:


emp_data.shape


# In[4]:


# Changing the names of the columns
emp_data.columns  = ['Id', 'Reason', 'Month', 'DOW', 'Season', 'Trans_exp', 'Distance', 'Service_time',
                     'Age', 'Avg_work_load', 'Hit_target', 'Disciplinary', 'Education', 'Children', 'Drink',
                     'Smoke', 'No_of_pet', 'Weight', 'Hieght', 'BMI', 'Hours_absent']


# In[5]:


emp_data.head()


# ## Pre - processing 
# ### Missing Value Analysis

# In[6]:


# Calculating missing percentage in the data
dict0 = {}
for i in emp_data.isna():
    dict0[i] = [i, (sum(emp_data.isna()[i]) * 100 / len(emp_data.Id))]
mv = pd.DataFrame(data = dict0)
mv = mv.drop([0])
mv = mv.T
mv.columns = ['Missing Values Percentage']
mv.sort_values(by='Missing Values Percentage', ascending=False)


# In[7]:


# Imputing the columns with missing values with mean or median
emp_data.Trans_exp = emp_data.Trans_exp.fillna(np.nanmean(emp_data.Trans_exp))
emp_data.Education = emp_data.Education.fillna(np.nanmedian(emp_data.Education))
emp_data.Disciplinary = emp_data.Disciplinary.fillna(np.nanmedian(emp_data.Disciplinary))
emp_data.Drink = emp_data.Drink.fillna(np.nanmedian(emp_data.Drink))
emp_data.Smoke = emp_data.Smoke.fillna(np.nanmedian(emp_data.Smoke))
emp_data.Age = emp_data.Age.fillna(np.nanmean(emp_data.Age))
emp_data.No_of_pet = emp_data.No_of_pet.fillna(np.nanmedian(emp_data.No_of_pet))
emp_data.Hours_absent = emp_data.Hours_absent.fillna(np.nanmedian(emp_data.Hours_absent))
emp_data.Distance = emp_data.Distance.fillna(np.nanmean(emp_data.Distance))
emp_data.Service_time = emp_data.Service_time.fillna(np.nanmean(emp_data.Service_time))
emp_data.Avg_work_load = emp_data.Avg_work_load.fillna(np.nanmean(emp_data.Avg_work_load))
emp_data.Reason = emp_data.Reason.fillna(np.nanmedian(emp_data.Reason))
emp_data.Hit_target = emp_data.Hit_target.fillna(np.nanmedian(emp_data.Hit_target))
emp_data.Month = emp_data.Month.fillna(np.nanmedian(emp_data.Month))
emp_data.Children = emp_data.Children.fillna(np.nanmedian(emp_data.Children))
emp_data.Weight = emp_data.Weight.fillna(np.nanmean(emp_data.Weight))
emp_data.Hieght = emp_data.Hieght.fillna(np.nanmean(emp_data.Hieght))


# In[8]:


# Calculating BMI from the given weights and hieghts
for i in range(emp_data.BMI.shape[0]):
    if np.isnan(emp_data.BMI[i]):
        emp_data.BMI[i] = emp_data.Weight[i] * 10000  /(emp_data.Hieght[i] ** 2)


# In[9]:


emp_data.count()


# ### Chi-squared Test for categorical features.

# In[10]:


names = ['Reason', 'Month', 'DOW', 'Season', 'Hit_target', 'Disciplinary', 'Education', 'Drink', 'Smoke']
for i in names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(emp_data['Hours_absent'], emp_data[i]))
    print(p)


# ### Correlation for numerical features.

# In[11]:


numerical = [i for i in emp_data if i not in names and i != 'Id']
Correlation = emp_data.loc[:,numerical].corr()    

f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(Correlation, mask=np.zeros_like(Correlation,dtype=np.bool), 
            cmap=sns.diverging_palette(220,100, as_cmap=True), 
            square=False, ax = ax, center=0, robust=True);


# ### Outlier Analysis

# In[12]:


plt.boxplot(emp_data.Hours_absent);
emp_data.Hours_absent.describe()


# In[13]:


q_3, q_1 = np.percentile(emp_data.Hours_absent, [75,25])
iqr = q_3 - q_1
minimum, maximum = np.percentile(emp_data.Hours_absent, 0), q_3 + (iqr * 1.5)

emp_data = emp_data.drop(emp_data[emp_data.Hours_absent > maximum].index)


# In[14]:


plt.boxplot(emp_data.Hours_absent);
emp_data.Hours_absent.describe()


# In[15]:


dict_normal = {}
for i in numerical:
    dict_normal[i] = [max(emp_data[i]), min(emp_data[i])]
    emp_data[i] = ((emp_data[i] - min(emp_data[i]))/(max(emp_data[i]) - min(emp_data[i])))


# In[16]:


dict_normal


# In[17]:


emp_data['Category_absent'] = pd.cut(emp_data.Hours_absent,  bins=4,labels= ['least', 'moderate', 'high', 'highest'])


# In[18]:


emp_data = emp_data.drop(['Distance', 'No_of_pet', 'Weight', 'Hieght', 'BMI', 'Hours_absent'], axis=1)


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(emp_data.iloc[:, 0:15], emp_data.iloc[:,15], test_size = 0.1)
C50_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
C50_model.fit(X_train, y_train)

predictions = C50_model.predict(X_test)

nn = C50_model.decision_path(X_test)
for i in [0,1,2,3]:
    print(nn.indices[nn.indptr[i] : nn.indptr[i+1]])


# In[69]:


score = C50_model.score(test.iloc[:,0:15], test.iloc[:,15])
score


# In[70]:


dot_file = open('ptdot', 'w')
ddd = tree.export_graphviz(C50_model, out_file=dot_file, feature_names=emp_data.columns[0:15], 
                     class_names=emp_data.columns[15], filled=True, leaves_parallel=True, rounded= True, 
                     special_characters=True)


# In[72]:


Source.from_file('ptdot')

