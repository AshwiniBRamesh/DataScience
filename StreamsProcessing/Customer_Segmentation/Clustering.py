#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats
import statsmodels.formula.api as sm

df = pd.read_csv(r"D:\BITs Pilani\third sem\pizza_customers.csv")
describe = df.describe()
df = df.drop(columns=['CustomerID'])

#Distribution Plot
di=sns.distplot(df['Annual Income (k$)'])
d2i=sns.distplot(df['Spending Score (1-100)'])
z=sns.boxplot(df['Annual Income (k$)'])
z=sns.boxplot(df['Spending Score (1-100)'])


# Removing Outliers
q75,q25=np.percentile(df.iloc[:,3],[75,25])
iqr=q75-q25
min=q25-(iqr*1.5)
max=q75+(iqr*1.5)
df=df.drop(df[df.iloc[:,3]<min].index)
df=df.drop(df[df.iloc[:,3]>max].index)


# Draw a scatter plot
df.plot.scatter(x = 'Age', y = 'Annual Income (k$)', s = 100);
df.plot.scatter(x = 'Age', y = 'Spending Score (1-100)', s = 100);
df.groupby(['Gender']).mean().plot(kind='pie', y='Spending Score (1-100)',autopct='%1.1f%%')
df.groupby(['Gender']).mean().plot(kind='pie', y='Age',autopct='%1.1f%%')




#K mean clustering- segmentation
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
from sklearn.cluster import KMeans

df['Gender']= label_encoder.fit_transform(df['Gender'])
X = df.iloc[:,[2,3]].values

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')


kmeans=KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans=kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


