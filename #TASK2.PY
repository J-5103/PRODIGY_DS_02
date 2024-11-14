import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

data = pd.read_csv(r'D:\Python\PRODIGY INFOTECH INTERNSHIP\TASK#2\titanic\train.csv')  

data.head()
data.info()

print(data.isnull().sum())

print(data.describe())

data['Age'].fillna(data['Age'].median(), inplace=True)

data.drop('Cabin', axis=1, inplace=True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

print(data.isnull().sum())

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

plt.figure(figsize=(8, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['Fare'], kde=True, color='green')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.xlabel('Sex (0=Male, 1=Female)')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', kde=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Fare vs. Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

plt.figure(figsize=(8, 6))
sns.catplot(x='Pclass', hue='Sex', col='Survived', data=data, kind='count')
plt.suptitle('Survival by Gender and Passenger Class')
plt.show()

sns.pairplot(data[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived', palette='coolwarm')
plt.suptitle('Pair Plot of Numerical Variables')
plt.show()