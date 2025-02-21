
import numpy as np #linear algebra
import pandas as pd #data processing​
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
import warnings
warnings.filterwarnings("ignore") #to ignore the warnings​
#for model building
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data.csv')
print("Total data points",data.shape[0])
print("Total number of features(as number of columns) are ", data.shape[1])

#CLEARING DATA
#Check for null values
data.isna().sum()
#Dropping 'ID an Unnamed; 32' Columns
data.drop(['id'], axis = 1 , inplace=True)
data.info()

#DATA PREPROCESSING
# counts of unique rows in the 'diagnosis' column
data['diagnosis'].value_counts()

plt.figure(figsize = (8,7))
sns.countplot(x="diagnosis", data=data, palette='magma')
#Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(25, 20))
sns.heatmap(correlation_matrix, annot=True, linewidths=0.3, linecolor="white", fmt=".2f", cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# Score > 0.85, strong correlation (Dark Blue)
# Score 0.4<>0.85, moderate correlation (between Dark blue and light blue)
# Score <0.4, weak correlation

#Splitting the data into train and test
# splitting data
X_train, X_test, y_train, y_test = train_test_split(data.drop('diagnosis', axis=1),
                                                    data['diagnosis'],test_size=0.3,random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)

#Standardize Data sets
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)


#DECISION TREE
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_train_tree = tree.predict(X_train)
predictions1 = tree.predict(X_test)
print("Confusion Matrix : \n",confusion_matrix(y_test, predictions1))
print("\nClassification Report:")
print(classification_report(y_true=y_test, y_pred=predictions1))
print("Accuracy of Decision Tree Classifier Model is:", accuracy_score(y_test, predictions1)*100,'%')


#KNN
error_rate = []

for i in range(1, 42):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 42), error_rate, color='purple', linestyle="--", marker='o', markersize=10, markerfacecolor='b')
plt.title('Error Rate vs K-value')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.show()

# From this graph, K value of 5,7 and 9 seem to show the lowest mean error. So using one of these values

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions2 = knn.predict(X_test)

print("Confusion Matrix : \n",confusion_matrix(y_test, predictions2))
print("\nClassification Report:")
print(classification_report(y_test, predictions2))

knn_model_acc = accuracy_score(y_test, predictions2)
print("Accuracy of K Neighbors Classifier Model is: ", knn_model_acc*100,'%')    

#SVM

svc_model = SVC(kernel="rbf")
svc_model.fit(X_train, y_train)
predictions3 = svc_model.predict(X_test)

print("Confusion Matrix: \n", confusion_matrix(y_test, predictions3))
print("\nClassification Report:")
print(classification_report(y_test, predictions3))

svm_acc = accuracy_score(y_test, predictions3)
print("Accuracy of SVM model is: ", svm_acc*100,'%')


# Select columns with correlation coefficient greater than 0.4 with respect to 'diagnosis'
high_corr_cols = correlation_matrix[correlation_matrix['diagnosis'] > 0.4].index.tolist()


# Remove 'diagnosis' from the list as it is the target variable
high_corr_cols.remove('diagnosis')

# Filter the dataframe based on selected columns
data_filtered = data[high_corr_cols + ['diagnosis']]

# Splitting the filtered data into train and test sets
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(data_filtered.drop('diagnosis', axis=1), data_filtered['diagnosis'], test_size=0.3, random_state=42)

print("Shape of training set with high correlation columns:", X_train_filtered.shape)
print("Shape of test set with high correlation columns:", X_test_filtered.shape)

# Standardize the filtered datasets
ss_filtered = StandardScaler()
X_train_filtered = ss_filtered.fit_transform(X_train_filtered)
X_test_filtered = ss_filtered.fit_transform(X_test_filtered)

# Decision Tree Classifier
tree_filtered = DecisionTreeClassifier(random_state=42)
tree_filtered.fit(X_train_filtered, y_train_filtered)

y_train_tree_filtered = tree_filtered.predict(X_train_filtered)
predictions1_filtered = tree_filtered.predict(X_test_filtered)
print("\nDecision Tree Classifier with High Correlation Columns:")
print("Confusion Matrix : \n",confusion_matrix(y_test_filtered, y_pred=predictions1_filtered))
print("Classification Report:")
print(classification_report(y_true=y_test_filtered, y_pred=predictions1_filtered))
print("Accuracy of Decision Tree Classifier Model with High Correlation Columns is:", accuracy_score(y_test_filtered, predictions1_filtered)*100,'%')

# KNN Classifier
error_rate_filtered = []

for i in range(1, 42):
    knn_filtered = KNeighborsClassifier(n_neighbors=i)
    knn_filtered.fit(X_train_filtered, y_train_filtered)
    pred_i_filtered = knn_filtered.predict(X_test_filtered)
    error_rate_filtered.append(np.mean(pred_i_filtered != y_test_filtered))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 42), error_rate_filtered, color='purple', linestyle="--", marker='o', markersize=10, markerfacecolor='b')
plt.title('Error Rate vs K-value (with High Correlation Columns)')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.show()

# From this graph, select the K value that shows the lowest mean error. So using one of these values

knn_filtered = KNeighborsClassifier(n_neighbors=5)
knn_filtered.fit(X_train_filtered, y_train_filtered)
predictions2_filtered = knn_filtered.predict(X_test_filtered)

print("\nK Neighbors Classifier with High Correlation Columns:")
print("Confusion Matrix: \n", confusion_matrix(y_test_filtered, predictions2_filtered))
print("\nClassification Report:")
print(classification_report(y_test_filtered, predictions2_filtered))

knn_model_acc_filtered = accuracy_score(y_test_filtered, predictions2_filtered)
print("Accuracy of K Neighbors Classifier Model with High Correlation Columns is: ", knn_model_acc_filtered*100,'%')

# SVC Classifier
svc_model_filtered = SVC(kernel="rbf")
svc_model_filtered.fit(X_train_filtered, y_train_filtered)
predictions3_filtered = svc_model_filtered.predict(X_test_filtered)

print("\nSVM Model with High Correlation Columns:")
print("Confusion Matrix: \n", confusion_matrix(y_test_filtered, predictions3_filtered))
print("\nClassification Report:")
print(classification_report(y_test_filtered, predictions3_filtered))

svm_acc_filtered = accuracy_score(y_test_filtered, predictions3_filtered)
print("Accuracy of SVM model with High Correlation Columns is: ", svm_acc_filtered*100,'%')


