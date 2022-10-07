import pandas as pd
df=pd.read_table('/content/Restaurant_Reviews (2).csv')
df
df['Review'].value_counts()
x = df['Review']
y=df['Liked'].values
x
y
import seaborn as sns
import matplotlib as plt
sns.distplot(df['Liked'],kde=False)
# import matplotlib.pyplot as plt
# plt.bar(x,y)
# plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=500)
x_train.shape
x_test.shape
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train_vect,y_train)
y_pred = model.predict(x_test_vect) 
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100
from sklearn.pipeline import make_pipeline
text_model = make_pipeline(CountVectorizer(),SVC())

text_model.fit(x_train,y_train)
y_pred = text_model.predict(x_test)
y_pred
# import joblib
# joblib.dump(text_model,'0-1')
# import joblib
# text_model= joblib.load('0-1')
text_model.predict(['This place is not worth your time, let alone Vegas'])