import numpy as np
import pandas as pd
import joblib

FILE_PATH = './heart_disease/pth/RandomForestClassifier.joblib'

feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
input = list(map(float,input('age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal 입력: ').split()))
X_test = pd.DataFrame([input],columns=feature_names)
model = joblib.load(FILE_PATH)
pred = model.predict(X_test)
if pred == 0: print('정상')
else: print('심장병 의심')
    
            
            
