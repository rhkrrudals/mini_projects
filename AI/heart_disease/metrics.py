import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score,consensus_score,f1_score,precision_recall_curve,roc_auc_score,roc_curve,classification_report
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

def get_clf_eval(y_test,pred,pred_proba):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test,pred_proba) 
    
    print('오차행렬')
    print(confusion)
    print(f'정확도 : {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f},F1: {f1:.4f}, AUC : {roc_auc:.4f}')
    
def get_eval_by_threshold(y_test,pred_proba_c1, thresholds):
    for custom_trheshold in thresholds:
        binaizer = Binarizer(thresholds=custom_trheshold).fit(pred_proba_c1)
        custom_predict = binaizer.fit(pred_proba_c1)
        print('임계값: ', custom_trheshold)
        get_clf_eval(y_test,custom_predict)
    
def roc_curve_plot(y_test,pred_proba_c1,class_name,FILE_PATH,name='img'):
    fprs,tprs,thresholds = roc_curve(y_test,pred_proba_c1)
    plt.title(f'{class_name} - AUC: {roc_auc_score(y_test,pred_proba_c1):.4f}')
    plt.plot(fprs,tprs,label='ROC')
    plt.plot([0,1],[0,1],'k--',label='Random')
    
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.xlabel('FPR( 1 - Specificity )'); plt.ylabel('TPR( Recall )')
    plt.legend(); plt.grid()
    plt.savefig(f'{FILE_PATH}{name}/{class_name}_roc_curve_plot')
    plt.show()