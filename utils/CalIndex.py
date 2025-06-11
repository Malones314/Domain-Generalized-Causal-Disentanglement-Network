from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, roc_curve, roc_auc_score, auc
import numpy as np
def cal_index(y_true, y_pred_labels, y_pred_probs):
    '''
    Calculate Accuracy, Recall, Precision, F1-Score
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    '''
    acc_ = accuracy_score(y_true, y_pred_labels)
    prec_ = precision_score(y_true, y_pred_labels, average='macro',labels=np.unique(y_pred_labels))
    recall_ = recall_score(y_true, y_pred_labels, average='macro',labels=np.unique(y_pred_labels))
    F1_score_ = f1_score(y_true, y_pred_labels, average='macro',labels=np.unique(y_pred_labels))
    # 计算AUC
    n_classes = y_pred_probs.shape[1]
    try:
        if n_classes > 2:
            auc_ = roc_auc_score(y_true, y_pred_probs, multi_class='ovo', average='macro')
        else:
            # 二分类时自动检测概率格式
            if y_pred_probs.shape[1] == 1:
                auc_ = roc_auc_score(y_true, y_pred_probs[:, 0])
            else:
                auc_ = roc_auc_score(y_true, y_pred_probs[:, 1])
    except ValueError:
        auc_ = -1.0
    return acc_, auc_, prec_, recall_, F1_score_
