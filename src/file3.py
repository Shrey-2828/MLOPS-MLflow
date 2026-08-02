import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub

df=load_breast_cancer()
x=pd.DataFrame(df.data,columns=df.feature_names)
y=pd.Series(df.target,name='target')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#params
grid={
    'max_depth': [None,3,5,7,9],
    'n_estimators':[10,50,100,150]
}

rf=RandomForestClassifier()
gd=GridSearchCV(estimator=rf,param_grid=grid,cv=5,n_jobs=-1)

#dagshub config
dagshub.init(repo_owner='Shrey-2828', repo_name='MLOPS-MLflow', mlflow=True)

#mlflow
mlflow.set_tracking_uri('https://dagshub.com/Shrey-2828/MLOPS-MLflow.mlflow')
mlflow.set_experiment("Breast cancer detection")

with mlflow.start_run() as parent:
    gd.fit(x_train,y_train)
    
    for i in range (len(gd.cv_results_['params'])):
        
        with mlflow.start_run(nested=True) as child: 
         mlflow.log_params(gd.cv_results_['params'][i])
         mlflow.log_metric("accuracy",gd.cv_results_["mean_test_score"][i])
    
    best_param=gd.best_params_
    best_score=gd.best_score_
    
    mlflow.set_tag('author','shrey patel')
    mlflow.log_artifact(__file__)
    mlflow.log_metric("accuracy",best_score)
    mlflow.sklearn.log_model(gd.best_estimator_ , 'Random forest')
    
    