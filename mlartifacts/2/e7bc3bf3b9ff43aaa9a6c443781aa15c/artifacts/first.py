import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


df=load_wine()
x=df.data
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#params
max_depth=3
estimators=8
mlflow.set_tracking_uri('http://127.0.0.1:5000')

with mlflow.start_run(experiment_id=2):
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=estimators,random_state=42,n_jobs=-1)
    rf.fit(x_train,y_train)
    
    y_pred=rf.predict(x_test)
    
    acc=accuracy_score(y_pred,y_test)
    pre=precision_score(y_pred,y_test,average='macro')
    cm=confusion_matrix(y_pred,y_test)
    
    plt.figure(figsize=(10,5))
    sns.heatmap(cm,annot=True,cmap='viridis',xticklabels=df.target_names,yticklabels=df.target_names)
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("confusion metrix")
    
    plt.savefig("confusion_matrix.png")
    
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",pre)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",estimators)
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    
    
    print(acc)
    



