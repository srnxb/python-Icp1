import pandas as pd

train_df= pd.read_csv('train.csv')
test_df= pd.read_csv('train.csv')
X_train= train_df.drop("Embarked",axis=1)
Y_train= train_df["Embarked"]
X_test= test_df.drop("PassengerId",axis=1).copy()
#print(train_df[train_df.isnull().any(axis=1)])
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
#train_df['Survived'] = train_df['Survived'].map({'Q': 1, 'S': 2, 'C':3}).astype(int)

#print(train_df['Sex'])
print(train_df['Survived'].corr(train_df['Sex']))