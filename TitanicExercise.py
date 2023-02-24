import pandas as pd

#Dataset is on the link below
df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv")

#We drop useless columns
df_n = df.drop(["Name","SibSp","Parch","Ticket","Cabin","Embarked","PassengerId"],axis="columns")
print(df_n.head())

meanAge = df_n.Age.mean()
df_n.Age = df_n.Age.fillna(meanAge)

#Converting string values in dataframe to numeric
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
df_n["Sex_n"] = le_sex.fit_transform(df_n["Sex"])

#We drop sex column because we don't need it anymore
df_final = df_n.drop(["Sex"],axis="columns")
print(df_final.head())

inputs = df_final.drop(["Survived"],axis="columns")
target = df_final.Survived
print(inputs.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.001)
print(X_train.head())

#Testing our model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs,target)

print(model.score(inputs,target))
print(X_test)
print(y_test)
print(model.predict([[3,16,39.6875,1]]))

print(model.score(X_train,y_train))

