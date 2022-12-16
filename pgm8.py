from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
iris=load_iris()
'''3 classes of iris: Setosa, versicolor, virginica
Number of Instances: 150 (50 in each of three classes)
Number of Attributes: 4 sepal length,sepal width,petal length,petal width in cm'''
iris_data=iris.data
iris_labels=iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_data,iris_labels,test_size=0.3)
kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train, y_train)
y_pred=kn.predict(X_test)
print('The confusion matrix is as follows:')
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))