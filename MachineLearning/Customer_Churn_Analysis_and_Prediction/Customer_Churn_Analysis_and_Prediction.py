# -*- coding: utf-8 -*-
"""
DM Assignment 1 - Group 166

"""

# Importing required libraries
import pandas as pd
import numpy as np

# Modules for data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Modules for Model Evaluation
    
from sklearn.metrics import accuracy_score, roc_curve 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score

# Libraries for data visualization
import seaborn as sn
import matplotlib.pyplot as plt

def dataPrep(file_name):
    # import the excel file in a data frame
    df= pd.read_csv(file_name)
    df.info()
    df.isna().any()
    # Column info shows TotalCharges to be of type object
    # Convert TotalCharges column to type float after replacing blanks with 0 as blank can't be converted to float
    
    df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    
    # Replace TotalCharges where nan with product of tenure and MonthlyCharges
    df['TotalCharges'] = np.where(np.isnan(df['TotalCharges']), df['tenure'] * df['MonthlyCharges'], df['TotalCharges'])
    # Look at statistical summary of numeric fields

    return df

def edaNumeric(df):
    df2 = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    plt.figure(figsize=(15, 12))
    plt.suptitle('Numerical Columns Distribution\n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
    for i in range(df2.shape[1]):
        plt.subplot(6, 3, i + 1)
        f = plt.gca()
        f.set_title(df2.columns.values[i])
        vals = np.size(df2.iloc[:, i].unique())
        if vals >= 100:
           vals = 100
        plt.hist(df2.iloc[:, i], bins=vals, color = '#ed838a')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
def edaCategorical(df):
    catVars = ['gender','SeniorCitizen', 'Partner','Dependents', 'PhoneService','MultipleLines',
    'InternetService','StreamingService', 'Contract', 'PaperlessBilling','PaymentMethod', 'Churn']
    fig, axes = plt.subplots(nrows = 3,ncols = 4,figsize = (24,18))
    plt.suptitle('\nDistribution for categorical variables\n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
       
    for i, item in enumerate(catVars):
        ax = df[item].value_counts().plot(kind = 'bar',ax=axes[int(i/4),int(i%4)], rot = 0, color ='lightblue' )       
        ax.set_title(item)


def churnCorrCheck(df):
    df2 = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
               'PhoneService', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
    correlations = df2.corrwith(df.Churn)
    correlations = correlations[correlations!=1]
    positive_correlations = correlations[
    correlations >0].sort_values(ascending = False)
    negative_correlations =correlations[
    correlations<0].sort_values(ascending = False)
    print('Positive Correlation with Churn: \n', positive_correlations)
    print('\nNegative Correlation with Churn: \n', negative_correlations)
    plt.figure(figsize=(15, 12))
    correlations = df2.corrwith(df.Churn)
    correlations = correlations[correlations!=1]
    correlations.plot.bar(figsize = (15,10), fontsize = 14, 
            color = 'lightblue', rot = 45, grid = True)
    
    plt.title('\n Correlation with Churn Rate \n',horizontalalignment="center", fontstyle = "normal", 
    fontsize = "22", fontfamily = "sans-serif")

# Function to calculate Variable Inflation Factors among attributes
def calc_vif(df):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    print(vif) 

def attributeCorrCheck(df):
        
    df2 = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService','PaperlessBilling','MonthlyCharges','TotalCharges']]
    print("\nVIF figures of original dataset\n")
    calc_vif(df2)
    
    # Check colinearity between MonthlyCharges and TotalCharges
        
    df2[['MonthlyCharges', 'TotalCharges']].plot.scatter(
    figsize = (12,8),x ='MonthlyCharges', y='TotalCharges', color =  'lightblue')
    plt.title('Co-relation between Monthly Charges and Total Charges \n',
    horizontalalignment="center", fontstyle = "normal", fontsize = "22", fontfamily = "sans-serif")
            
    # Check VIF again'
    df2 = df2.drop(columns = "TotalCharges")
    #Revalidate Colinearity:
    df2 = df[['gender','SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'PaperlessBilling','MonthlyCharges']]
    print("\nVIF figures after dropping related variables\n")
    calc_vif(df2)
    
    # Remove TotalCharges from original data frame
    dataset = df.drop(columns = "TotalCharges")
    return dataset

def evalModel (trainX, trainY, testY, predY):
    # Evaluate Model performance on Test Set:
    acc = accuracy_score(testY, predY)
    prec = precision_score(testY, predY)
    rec = recall_score(testY, predY)
    f1 = f1_score(testY, predY)
    f2 = fbeta_score(testY, predY, beta=2.0)
    modelScores = pd.DataFrame([[acc, prec, rec, f1, f2]],columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score'])
    print(modelScores)
    
    accuracies = cross_val_score(estimator = classifier, X = trainX, y = trainY, cv = 10)
    print("\nAccuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(),accuracies.std() * 2))
        
    classifier.fit(trainX, trainY) 
    probs = classifier.predict_proba(testX) 
    probs = probs[:, 1] 
    classifier_roc_auc = accuracy_score(testY, predY )
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(testY, classifier.predict_proba(testX)[:,1])
    plt.figure(figsize=(14, 6))
    
    # ROC Curve
    plt.plot(rf_fpr, rf_tpr, 
    label='Logistic Regression Area %0.2f)' % classifier_roc_auc)
    # Plot Base Rate ROC
    plt.plot([0,1], [0,1],label='Base Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate \n',horizontalalignment="center",
    fontstyle = "normal", fontsize = "medium", 
    fontfamily = "sans-serif")
    plt.xlabel('\nFalse Positive Rate \n',horizontalalignment="center", fontstyle = "normal", fontsize = "medium", 
    fontfamily = "sans-serif")
    plt.title('ROC Curve \n',horizontalalignment="center", fontstyle = "normal", fontsize = "22", 
    fontfamily = "sans-serif")
    plt.legend(loc="lower right", fontsize = "medium")
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.yticks(rotation=0, horizontalalignment="right")
    plt.show()    
    
def churnPred(trainX, trainY, testX)    :
    lr = LogisticRegression(random_state = 0, penalty = 'l2')
    lr.fit(trainX, trainY)
    
    predY = lr.predict(testX)
    # calculate churn probability
    predY_prob_score = lr.predict_proba(testX)
    predY_prob_score  = predY_prob_score[:, 1]
    
    churnProbability = pd.concat([testCustID, testY], axis = 1).dropna()
    churnProbability['ChurnPrediction'] = predY
    churnProbability["ChurnProbability(%)"] = predY_prob_score
    churnProbability["ChurnProbability(%)"] = churnProbability["ChurnProbability(%)"]*100
    churnProbability["ChurnProbability(%)"]= churnProbability["ChurnProbability(%)"].round(2)
    churnProbability = churnProbability[['customerID', 'Churn', 'ChurnPrediction', 'ChurnProbability(%)']]    
    return churnProbability

""" Main Program Begins  """

file_name = 'dataset.csv'

print("Phase 1: Read File, Check Data Profile and Data Clean-up\n")
df = dataPrep(file_name)

print("\nPhase 2: Exploratory Data Analysis")
print("\nPhase 2.1: Data Spread For Numeric Columns (check plot pane)")
edaNumeric(df)

color = sn.color_palette()
print("\nPhase 2.2: Distribution of data for categorical variables (check plot pane)")
edaCategorical(df)


# label encode categorical independent variables with 2 unique values
le = LabelEncoder()
print("\nPhase3: Label Encoding")
le_count = 0
for cols in df.columns[1:]:
    if df[cols].dtype == 'object':
        if len(list(df[cols].unique())) <= 2:
            le.fit(df[cols])
            df[cols] = le.transform(df[cols])
            le_count += 1
            
print("\nPhase 4.1: Correlation of independent variable with prediction variable\n")
churnCorrCheck(df)

print("\nPhase 4.2: Check co-relation among variables and achieve dimentionality reduction")
df = attributeCorrCheck(df)


# remove customerID from dataset:
custID = df["customerID"]
df = df.drop(columns="customerID")
# Convert rest of categorical variable into indicator variables
df= pd.get_dummies(df)
# Add customerID again:
df = pd.concat([df, custID], axis = 1)

# Drop classification label column before splitting data into training and test sets
label = df["Churn"]
df = df.drop(columns="Churn") 

print("\nPhase 5: Split into training and test data")
trainX, testX, trainY, testY = train_test_split(df, label,stratify=label, test_size = 0.2, random_state = 0)
trainCustID = trainX['customerID']
trainX = trainX.drop(columns = ['customerID'])
testCustID = testX['customerID']
testX = testX.drop(columns = ['customerID'])

print("\nPhase 6: Feature scaling")
x_scale = StandardScaler()
trainX2 = pd.DataFrame(x_scale.fit_transform(trainX))
trainX2.columns = trainX.columns.values
trainX2.index = trainX.index.values
trainX = trainX2
testX2 = pd.DataFrame(x_scale.transform(testX))
testX2.columns = testX.columns.values
testX2.index = testX.index.values
testX = testX2

print("\nPhase 7: Train Logistic Regression model & predict on test data")
classifier = LogisticRegression(random_state = 0, penalty = "l2")
classifier.fit(trainX, trainY)
# Predict the Test data
predY = classifier.predict(testX)

print("\nPhase 8: Evaluate model performance (Logistic Regression)")
evalModel(trainX, trainY, testY, predY)

print("\nPhase 9: Summary of Churn prediction and churn probability score for each customer")
churnProbability = churnPred(trainX,trainY,testX)
print(churnProbability)



