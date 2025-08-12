import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

data=pd.read_csv('ml4.csv')
df=pd.DataFrame(data)

x=df[['study_hours','Attendance']]
y=df['result']
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(x,y)
plt.figure(figsize=(8,6))
plot_tree(dtc,feature_names=['study_hours','Attendance'],class_names=['0','1'],filled=True)
plt.show()

new=[[5,85]]
pred=dtc.predict(new)
print("Prediction for new student:","1" if pred[0]==1 else "0")
