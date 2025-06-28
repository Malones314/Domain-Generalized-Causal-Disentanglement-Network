def cal_index(y_true, y_pred_labels, y_pred_probs):
    """
    Calculate Accuracy, Recall, Precision, F1-Score, and AUC.
    This corrected version removes the flawed `labels` parameter to ensure accurate metric calculation.
    """
    acc_ = accuracy_score(y_true, y_pred_labels)

    # 添加 `zero_division=0` 防止因分母为零发出警告
    prec_ = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
    recall_ = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
    F1_score_ = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)

    # 计算 AUC (此部分逻辑正确)
    n_classes = y_pred_probs.shape[1]
    try:
        # 确保 y_true 中有多个类别才能计算AUC
        if len(np.unique(y_true)) < 2:
            raise ValueError("Only one class present in y_true. ROC AUC score is not defined.")

        if n_classes > 2:
            auc_ = roc_auc_score(y_true, y_pred_probs, multi_class='ovo', average='macro')
        else:
            # 对于二分类，标准做法是使用正类（标签为1）的概率
            auc_ = roc_auc_score(y_true, y_pred_probs[:, 1])

    except ValueError as e:
        # 打印错误信息以帮助调试，然后返回一个表示失败的值
        # print(f"AUC calculation failed: {e}")
        auc_ = -1.0

    return acc_, auc_, prec_, recall_, F1_score_
