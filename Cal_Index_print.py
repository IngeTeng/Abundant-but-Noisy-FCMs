import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from scipy import interp
from sklearn.metrics import auc

#SS_Mean
def cal_SS_Mean(BestNN,NN):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(NN.shape[0]):
        for j in range(NN.shape[1]):
            if np.abs(BestNN[i,j]) > 0.05 and np.abs(NN[i,j]) > 0.05:
                tn = tn + 1
            elif np.abs(BestNN[i,j]) <= 0.05 and np.abs(NN[i,j]) <= 0.05:
                tp = tp + 1
            elif np.abs(BestNN[i,j]) > 0.05 and np.abs(NN[i,j]) <= 0.05:
                fn = fn + 1
            else:
                fp = fp + 1
    specificity = tp/(tp+fn)
    sensitivity = tn/(tn+fp)
    ss_mean = 2 * specificity * sensitivity / (specificity+sensitivity)
    return ss_mean


#AUC(macro)
def cal_AUC_macro(label, y_score,  n_classes = 3):

    #label=[-1,0,1]
    #y_score={[0.2,0,0.8],[],[]}每个元素对三个标签的归属度
    # one vs rest方式计算每个类别的TPR/FPR以及AUC
    label = label_binarize(label, classes=[-1, 0, 1])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        a = label[:, i]
        b =  y_score[:, i]
        fpr[i], tpr[i], _ = roc_curve(label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # 微平均方式计算TPR/FPR，最后得到AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # AUC = roc_auc["micro"]

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    AUC = auc(fpr["macro"], tpr["macro"])
    return AUC,fpr,tpr,roc_auc

def cal_index_print(TrueNN, estimate_FCM, candidate_FCM, Q):

    candidate_FCM_SS_Mean = []
    candidate_FCM_AUC = []

    sum = 0
    sum_AUC = 0

    avge = 0
    avge_AUC = 0

    num = 0

    numRow = candidate_FCM.shape[1]
    numCol = candidate_FCM.shape[2]
    # Calulate
    for i in range(candidate_FCM.shape[0]):
        aj = candidate_FCM[i]
        ssmean = cal_SS_Mean(candidate_FCM[i], TrueNN)
        for m in range(aj.shape[0]):
            for n in range(aj.shape[1]):
                if aj[m][n] > 0.05:
                    aj[m][n] = 1
                elif aj[m][n] < -0.05:
                    aj[m][n] = -1
                else:
                    aj[m][n] = 0

        tp = np.zeros(shape=(TrueNN.shape[0], TrueNN.shape[0]))
        for iii in range(TrueNN.shape[0]):
            for jjj in range(TrueNN.shape[1]):
                p = TrueNN[iii][jjj]
                if TrueNN[iii][jjj] < -0.05:
                    tp[iii][jjj] = -1
                if TrueNN[iii][jjj] > 0.05:
                    tp[iii][jjj] = 1
                if TrueNN[iii][jjj] < 0.05 and TrueNN[iii][jjj] > -0.05:
                    tp[iii][jjj] = 0
        # True Vector
        truelabel = tp.reshape(-1, 1)

        y_score = []
        for ii in range(numRow):
            for jj in range(numCol):
                tmp = []
                v = candidate_FCM[i][ii][jj]
                res_v = 1 - abs(v)
                if v < 0:
                    tmp = [abs(v), res_v / 2, res_v / 2]
                if v == 0:
                    tmp = [0, 1, 0]
                if v > 0:
                    tmp = [res_v / 2, res_v / 2, abs(v)]
                y_score.append(tmp)
        y_score = np.array(y_score)
        AUC, _, _, _ = cal_AUC_macro(truelabel, y_score)

        candidate_FCM_AUC.append(AUC)
        candidate_FCM_SS_Mean.append(ssmean)

        num = num + 1
        sum = sum + ssmean
        sum_AUC = sum_AUC + AUC

    # Average
    avge = sum / num
    avge_AUC = sum_AUC / num

    # STD
    STD = np.std(candidate_FCM_SS_Mean)
    STDAUC = np.std(candidate_FCM_AUC)

    # SS_Mean
    estimate_SS_Mean = cal_SS_Mean(estimate_FCM, TrueNN)

    # AUC
    y_pred = []
    for i in range(numRow):
        for j in range(numCol):
            tmp = [Q[0, i, j], Q[1, i, j], Q[2, i, j]]
            y_pred.append(tmp)
    y_pred = np.array(y_pred)
    AUC_micro, fpr, tpr, roc_auc = cal_AUC_macro(truelabel, y_pred)

    print("==========AUC_Macro==========")
    print('candidate_FCM_AUC_Macro:', candidate_FCM_AUC)
    print('estimate_AUC_Macro:', AUC_micro)
    print('avge AUC_Macro:', avge_AUC)
    print('Std AUC:', STDAUC)
    print("\n")
    print("==========SS_Mean==========")
    print('candidate_FCM_SS_Mean:', candidate_FCM_SS_Mean)
    print('estimate_SS_Mean:', estimate_SS_Mean)
    print('avge SS_Mean:', avge)
    print('Std SS_Mean:', STD)

