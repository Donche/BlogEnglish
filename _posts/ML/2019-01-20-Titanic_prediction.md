---
layout: post
title: Titanic - Kaggle Competition
header-style: text
category: Machine Learning
catalog: true
mathjax: true
tags: 
    - 2019
    - Python
    - Machine Learning
    - Kaggle
---

*It was the ship of dreams to everyone else.*

# 0. Introduction

The Titanic sinking accident was a famous shipwreck in the North Atlantic from late night on April 14, 1912 to early morning on the 15th. The disaster shocked the world, killing more than 1,500 people and becoming the worst shipwreck in peacetime in history[[1]](#1).   
**Titanic: Machine Learning from Disaster**  is an entry-level competition for kaggle and is currently the most contested team with more than 10,000 teams. After the launch of this competition, many teams participated and achieved good results. There are many analytical methods worth learning in the kernels. I spent some time analyzing the data and predicting the survival of the passengers and learned a lot.    
One of the great advantages of the dataset on Kaggle is that it's not as straightforward as the ideal perfect data, and the data cleaning isn't too complicated. It is like an excessive phase of ideal data sets and real-world data.   
So this blog is to record how I analyze this data set and predict the survival of passengers.  

# 1. Problem Definition

The Titanic Dataset is a very good dataset for begineers in data science and participate in competitions in Kaggle. The purpose of this data set is to predict the survival of passengers on the Titanic with known personal information.  
For the training set, we know the economic and social status (Pclass), name, gender, age, spouse and siblings (SibSp), parent and child number (parch),  ticket number, ticket, fare,cabin and boarding dock (Embarked) of a total of 891 passengers.  about 177 people do not have age information, 687 people do not have cabin information, 2 people do not have boarding dock information. The goal is based on this information The build model predicts the survival of the test set passengers. The test set has a total of 418 passenger information, of which 86 have no age information, 1 has no fare information, and 327 have no cabin information.    

# 2. Exploratory Data Analysis   

## 2.1 Overview   
Let's import the dataset and see what do we have:    
```python
import pandas as pd
raw_data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data = raw_data.set_index('PassengerId')
data_val = data_test.set_index('PassengerId')

data.head()``	
```

| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
| --- | --- | --- |
| 1 | 0 | 3 | Braund, Mr. Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 2 | 1 | 1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 STON/O2. 3101282 | 7.9250 | NaN | S |
| 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 53.1000 | C123 | S |
| 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |

```python
data.info()
```
><class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 1 to 891
Data columns (total 11 columns):
Survived    891 non-null int64
Pclass      891 non-null int64
Name        891 non-null object
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Ticket      891 non-null object
Fare        891 non-null float64
Cabin       204 non-null object
Embarked    889 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 83.5+ KB   
```python
data_val()
```
><class 'pandas.core.frame.DataFrame'>
Int64Index: 418 entries, 892 to 1309
Data columns (total 10 columns):
Pclass      418 non-null int64
Name        418 non-null object
Sex         418 non-null object
Age         332 non-null float64
SibSp       418 non-null int64
Parch       418 non-null int64
Ticket      418 non-null object
Fare        417 non-null float64
Cabin       91 non-null object
Embarked    418 non-null object
dtypes: float64(2), int64(3), object(5)
memory usage: 35.9+ KB   

As we can see, many people’s cabin information is missing, as well as age and fare. However, these could be important (Cabin position may be an important factor in the success of escape, also the famous code: women and children first). 

## 2.2 Analysis

### 2.2.1 Sex

We plot the survival rate by sex. It clearly shows that women have a survival probability that is about three times that of men.

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_Cabin.png?raw=true)
### 2.2.2 Pclass

Pclass means the ticket class. This is an important information to indicate the social status of a person. In most cases, rich people will have more privileges to escape. For example, better cabin position, more lifesaving devices, I guess. Whatever the reason, the survival rate of first-class people is undoubtedly higher than that of others. And as always, even a male in 1st class has less chance to survived than a female in 3rd class. This shows that ticket class is also an important feature.

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_pclass.png?raw=true)

### 2.2.3 Age

If we stick to the code of conduction : women and children first, we should know that age must also be a powerful feature that would definitely affect a person's survival. As age can be a continuous variable (which in fact is the case), binning data will help to indicate trends of survival rate. The numbers of people and survivors of all ages are shown below. We notice that even there is a large number of people between the ages of 16 and 40, they didn't survive a lot. Instead, many kids survived even though they are much more vulnerable.   

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_age.png?raw=true)

### 2.2.4 SibSp and Parch

SibSp means the number of siblings / spouses aboard the Titanic and Parch means the number of of parents / children aboard the Titanic. In theory, it is easier for a family to unite and survive together (although women and children have a greater probability of survival). We do notice that a certain number of family members has more chance to survive, this may be helpful.    

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_SibSp.png?raw=true)

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_Parch.png?raw=true)

### 2.2.5 Fare

Fare is also an important indicator of a person’s social status. We found that passengers with fare greater than 140 basically survived while the cheapest fare has the lowest survival rate.

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_fare.png?raw=true)

### 2.2.6 Embarked

The embarked port has also a significant influence on survival rate. People who embarked on Cherbourg have somehow more chance to survive.

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_embarked.png?raw=true)

### 2.2.7 Cabin

The cabin also has an influence on the survival rate. People in the cabin closer to the exit (or better position) (maybe) can get to the lifeboats faster. As we have too much information of cabin lost, it's hard to predict or fill in with a reasonable value. Thus `U` is used as `unknown cabin`. 

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_Cabin.png?raw=true)

# 3. Data Cleaning

Data cleaning is an important step to make data more comprehensible to classifier. All the empty information needs to be filled and continuous data to be binned. As the classifier won't be processing strings, all the string data needs to be converted to labels with `LabelEncoder`.

## 3.1 cabin, embarked, fare and familysize

Firstly we combine the train and test data together to fill all the missed information.   

Cabin is filled as we mentioned before. As for Embarked, we have only 2 values lost in the training data, so we set them as 'S' which is the most frequent value for Embarked. There's also one `Fare` value lost, which is set as the median fare of all the passengers.

An extra information is added as well. We observed a certain pattern (not so obvious) in `Parch` and `SibSp`, so another feature `Familysize` is considered as the combination of the two. And this time, it become much more evident:

![](https://github.com/Donche/en/blob/master/_posts/ML/img/Titanic_Family.png?raw=true)

## 3.2 Title

Another important information is the title of a person, which is a symbol of a person's social status, age, marital status, and even position on board. The title of a person is extracted from its name with following code.

```python
replacement = {
    'Mr': 1,
    'Sir': 1,
    'Don': 1,
    'Rev': 1,
    'Jonkheer': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Capt': 3,
    'Master': 4,
    'Miss': 5,
    'Mlle': 5,
    'Mrs': 6,
    'Mme': 6,
    'Ms': 6,
    'Lady': 6,
    'Dona': 6,
    'the Countess': 6
}

data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data_val['Title'] = data_val['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

data['Title'] = data['Title'].apply(lambda x: replacement.get(x))
data_val['Title'] = data_val['Title'].apply(lambda x: replacement.get(x))
```

## 3.3 Surname

The surname can be easy to distinguish is two people is from the same family. We have certainly found that people without family members are more likely to die than those with 3. So we note family with their surname, and `'alone'` if a person is alone on board. This idea is inspired by a kernel in Kaggle using only names to predict survival[[4]](#4).

```python
#get surname
data['Surname'] = data['Name'].str.split(", ", expand=True)[0]
data_val['Surname'] = data_val['Name'].str.split(", ", expand=True)[0]
#combine all the data
surname_combined = data.append(data_val)
surname_combined.reset_index(inplace=True)
#find family numbers
surname_combined_grouped = surname_combined.groupby('Surname')
surname_family_filtered = surname_combined_grouped.filter(lambda x : len(x) > 1 )
data['Surname_unique'] = surname_family_filtered['Surname']
data['Surname_unique'].fillna('alone',  inplace = True)
data_val['Surname_unique'] = surname_family_filtered['Surname']
data_val['Surname_unique'].fillna('alone',  inplace = True)
```



## 3.4  Age

About 177 people don't have age information. But we still can make a guess according to the sex, title and class of a person. We use median age of each condition as the age to each person.

```python
#combine train data and test data
combined = data.append(data_val)
combined.reset_index(inplace=True)
combined.drop([ 'PassengerId'], inplace=True, axis=1)

age_median = combined.groupby(['Sex','Pclass','Title']).median()
age_median = age_median.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
#return median value of each condition
def fill_age(row):
    condition = (
        (age_median['Sex'] == row['Sex']) & 
        (age_median['Title'] == row['Title']) & 
        (age_median['Pclass'] == row['Pclass'])
    ) 
    return age_median[condition]['Age'].values[0]
#apply the function
data['Age'] = data.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
data_val['Age'] = data_val.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
#fill the rest info with the median if we don't have the corresponding condition
data['Age'].fillna(combined['Age'].median(), inplace = True)
data_val['Age'].fillna(combined['Age'].median(), inplace = True)
```

Then, the ages are binned into four groups according to the quartile, which is in fact really important to the result. It has increased the public score from 0.76 to 0.8, a very huge improvement. It is also easy to understand. The pattern for grouping people by age are much more obvious than those directly from age.

```python
data.loc[data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64, 'Age'] = 4


data_val.loc[data_val['Age'] <= 16, 'Age'] = 0
data_val.loc[(data_val['Age'] > 16) & (data_val['Age'] <= 32), 'Age'] = 1
data_val.loc[(data_val['Age'] > 32) & (data_val['Age'] <= 48), 'Age'] = 2
data_val.loc[(data_val['Age'] > 48) & (data_val['Age'] <= 64), 'Age'] = 3
data_val.loc[data_val['Age'] > 64, 'Age'] = 4
```

## 3.5 Fare

Fare is also binned to improve the performance of the classifier (and it really does):

```python
data.loc[data['Fare'] <= 7.9, 'Fare'] = 0
data.loc[(data['Fare'] > 7.9) & (data['Fare'] <= 14.45), 'Fare'] = 1
data.loc[(data['Fare'] > 14.45) & (data['Fare'] <= 31.5), 'Fare'] = 2
data.loc[data['Fare'] > 31.5, 'Fare'] = 3


data_val.loc[data_val['Fare'] <= 7.9, 'Fare'] = 0
data_val.loc[(data_val['Fare'] > 7.9) & (data_val['Fare'] <= 14.45), 'Fare'] = 1
data_val.loc[(data_val['Fare'] > 14.45) & (data_val['Fare'] <= 31.5), 'Fare'] = 2
data_val.loc[data_val['Fare'] > 31.5, 'Fare'] = 3
```

# 4. Modeling and Predicting

## 4.1 Feature selection

Now, 11 features are already available, some of which are highly relevant to the survival of passengers, while others are not. So, in this part, we need to find out which features are more irrelevant to the results and will be removed in subsequent process.    

We use random forest as the default algorithm. After training all the data, feature importance can be shown with the following code:    

```python
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=False, inplace=True)
features.set_index('feature', inplace=True)
print(features)
```

And the result is as follows:   

```
                     importance
feature                        
Title                  0.229910
Surname_unique_code    0.190721
Sex_code               0.128205
Pclass                 0.099802
Cabin_code             0.075117
Fare                   0.061674
Age                    0.048674
SibSp                  0.047281
Embarked_code          0.046067
FamilySize             0.043182
Parch                  0.029366
```

`parch` is clearly the most unimportant feature, followed by `Age`, `SibSp`, `Embarked_code` and `FamilySIze`. So `Parch` will be removed in the rest of the analysis.    

## 4.2 Model selection

We have a list of potential choices as classifier: Random Forest, Decision Tree, SVM, XGBoost, etc. For each classifier, scores on test set have been estimated in order to choose the most suitable for this dataset. I have decided to use several different classifiers make the prediction at the same time and vote for the final result.     

So I've chosen Random Forest, XGBoost and SVM as classifiers.

## 4.3 Parameter tuning

Then, during this step, parameter tuning using cross validation. For example, the parameter tuning for random forest is as follows:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(n_jobs= -1)
parameter_grid = {'n_estimators': [100,250,500,750],
                  'max_features' : [2,3,4,5], 
                  'max_depth' : [3,4,5],
                  'criterion' : ['gini', 'entropy']
                 }
cross_validation = StratifiedKFold(n_splits =5)
grid_search = GridSearchCV(clf,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(data_calc, target)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
```

And the result:

```
Best score: 0.8327721661054994
Best parameters: {'criterion':'gini', 'max_depth': 4, 'max_features': 3, 'n_estimators': 500}
```

## 4.4 Predict

After parameter tuning for each classifier, the result of the test data is voted according to the result of each classifier. The it's been written in to a `.csv` file

```python
y_final = list(map(int,(y_rf + y_xgb + y_svm)/3+0.5))
submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": y_final
    })

submission.to_csv("result.csv",index=False)
```

# 5. Result and Conclusion

The Public score of the dataset using this result is 0.80861 and it's top 9% of the competition.

The result is really good and it's actually surprising to get more than 80% accuracy with just this information. But this is just a Kaggle competition, the real world data analysis can be more complicated and more difficult to handle.

Just as a bridge between theory and reality, we can still have fun with it !

*reference materials*

1. <span id="1"></span>[Sinking of the RMS Titanic - wikipedia](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic)
2. [A Statistical Analysis & ML workflow of Titanic - Kaggle](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic)
3. [A Comprehensive ML Workflow with Python - Kaggle](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
4. <span id="4"></span>[Titanic using Name only - Kaggle](https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818)