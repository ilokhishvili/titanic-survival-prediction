import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


os.system('cls')
df = pd.read_csv(r'C:\Users\Mariami\Desktop\Phyton Lessons\phpMYEkMl.csv')

# ##print(df.describe)
# df.replace("?", np.nan, inplace=True)
# df["age"] = df["age"].astype(float)
# df["age"].fillna(df["age"].mean(), inplace= True)

# df['fare'] = df['fare'].astype(float)
# df['fare'].fillna(df['fare'].mean(), inplace=True)


# df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# df.drop(['cabin', 'body', 'boat'], axis=1, inplace=True)



# print("-----")
# ##print(df.isnull().sum())



# df['sex'].value_counts().plot(kind='bar')
# plt.title('Sex Distribution')
# plt.xlabel('Sex')
# plt.ylabel('Count')
# plt.show()


# df['survived'].value_counts().plot(kind='bar', color=['red', 'green'])
# plt.title('Survival Distribution')
# plt.xlabel('Survived (0=No, 1=Yes)')
# plt.ylabel('Count')
# plt.show()


# df['age'].plot(kind='hist', bins=20, edgecolor='black')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.show()


# le = LabelEncoder()
# df['sex'] = le.fit_transform(df['sex'])
# df['embarked'] = le.fit_transform(df['embarked'])


# X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
# y = df['survived']


# # გავყოთ მონაცემები: 80% სწავლა, 20% ტესტი
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# შევქმნათ მოდელი
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# ვიწინასწარმეტყველოთ
#y_pred = model.predict(X_test)

# ვნახოთ სიზუსტე
#print("Accuracy:", accuracy_score(y_test, y_pred))

print(df.isnull().sum())
# df.replace("?", np.nan, inplace=True)
print("---")
df.replace("?", np.nan, inplace= True)