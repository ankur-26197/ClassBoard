# -*- coding: utf-8 -*-
"""
@author: ANKUR SINGH
"""
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

st.title("ClassBoard")
dataset_name=st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer","Wine","Digits"))
classifier_name=st.sidebar.selectbox("Select Classifier", ("Logistic Regression","SVM","Random Forest","KNN"))

def main():
    def get_dataset(dataset_name):
        if dataset_name=="Iris":
            data=datasets.load_iris()
        elif dataset_name=="Breast Cancer":
            data=datasets.load_breast_cancer()
        elif dataset_name=="Wine":
            data=datasets.load_wine()
        elif dataset_name=="Digits":
            data=datasets.load_digits()
        
        X=data.data
        Y=data.target
        
        return X,Y
    
    def plot_metrics(metrics):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        
        if "Confusion Matrix" in metrics:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,X_test,Y_test)
            st.pyplot()
        
        if "ROC curve" in metrics:
            st.subheader("ROC Curve")
            plot_roc_curve(model,X_test, Y_test)
            st.pyplot()
        
        if "Precision-Recall Curve" in metrics:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,X_test, Y_test)
            st.pyplot()
            
    def model_maker(parameters):
        if classifier_name=="SVM":
            st.subheader("Support Vector Machine results")
            model=SVC(C=parameters["c"],kernel=parameters["kernel"],gamma=parameters["gamma"])
        
        elif classifier_name=="Logistic Regression":
            st.subheader("Logistic Regression results")
            model=LogisticRegression(C=parameters["c"],max_iter=parameters["epochs"])
            
        elif classifier_name=="Random Forest":
            st.subheader("Random Forest results")
            model=RandomForestClassifier(n_estimators=parameters["n_estimators"],max_depth=parameters["max_depth"],bootstrap=parameters["bootstrap"])
        
        elif classifier_name=="KNN":
            st.subheader("KNN results")
            model=KNeighborsClassifier(n_neighbors=parameters["neighbours"])
        
            
        model.fit(X_train,Y_train)
        accuracy=model.score(X_test,Y_test)
        Y_pred=model.predict(X_test)
        st.write("Accuracy: ",accuracy.round(2))
        st.write("Precision: ",precision_score(Y_test,Y_pred, average='macro').round(2))
        st.write("Recall: ", recall_score(Y_test, Y_pred, average='macro').round(2))
        
        return model
    
    def metrics_option_giver():
        if len(np.unique(Y))==2:
            t=("Confusion Matrix","ROC curve","Precision-Recall Curve")
        
        t=("Confusion Matrix")
        return t
    X,Y=get_dataset(dataset_name)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3,random_state=0)
    
    if classifier_name=="SVM":
        st.sidebar.subheader("Model hyperparamters")
        c=st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel=st.sidebar.radio("Kernel",("rbf","linear"), key="kernel")
        gamma=st.sidebar.radio("Gamma",("scale","auto"),key="gamma")
        metrics=st.sidebar.multiselect("Select metrics for plotting", ("Confusion Matrix","ROC curve","Precision-Recall Curve") if len(np.unique(Y))==2 else ("Confusion Matrix",))
        parameters={"type":"SVM","c":c,"kernel":kernel,"gamma":gamma,"metrics":metrics}
        
    
    if classifier_name=="Logistic Regression":
        st.sidebar.subheader("Model hyperparameters")
        c=st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C')
        epochs=st.sidebar.slider("Epochs",1,500,key="epochs")
        metrics=st.sidebar.multiselect("Select metrics for plotting", ("Confusion Matrix","ROC curve","Precision-Recall Curve") if len(np.unique(Y))==2 else ("Confusion Matrix",))
        parameters={"type":"logistic","c":c,"epochs":epochs,"metrics":metrics}
        
    if classifier_name=="Random Forest":
        st.sidebar.subheader("Model hyperparameters")
        n_estimators=st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
        max_depth=st.sidebar.number_input("Depth of tree",1,20,step=1,key="max_depth")
        bootstrap=st.sidebar.radio("Bootstrap samples while building trees",("True","False"),key="boostrap")
        metrics=st.sidebar.multiselect("Select metrics for plotting", ("Confusion Matrix","ROC curve","Precision-Recall Curve") if len(np.unique(Y))==2 else ("Confusion Matrix",))
        parameters={"type":"random","n_estimators":n_estimators,"max_depth":max_depth,"bootstrap":bootstrap,"metrics":metrics}
     
    if classifier_name=="KNN":
        st.sidebar.subheader("Model hyperparamters")
        neighbours=st.sidebar.slider("Number of neighbours (K)",1,5,key="neighbours")
        metrics=st.sidebar.multiselect("Select metrics for plotting", ("Confusion Matrix","ROC curve","Precision-Recall Curve") if len(np.unique(Y))==2 else ("Confusion Matrix",))
        parameters={"neighbours":neighbours,"metrics":metrics}
     
        
    if st.sidebar.button("Classify",key="classify"):
        model=model_maker(parameters)
        plot_metrics(metrics)
            

if __name__ == '__main__':
    main()