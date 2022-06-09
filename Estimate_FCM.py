import numpy as np
import membership_function as mf
import Bayesian_optimization as bo

def cal_estimate_FCM(candidate_FCM):
    numFCM = candidate_FCM.shape[0]
    numRow = candidate_FCM.shape[1]
    numCol = candidate_FCM.shape[2]

    E = np.zeros(shape=(3, numRow, numCol))
    E_Res = np.zeros(shape=(3, numRow, numCol))

    tmp = np.zeros(shape=(3))
    for n in range(numFCM):  # The number of candidate FCMs
        for i in range(numRow):
            for j in range(numCol):
                # spare
                if np.abs(candidate_FCM[n, i, j]) < 0.05:
                    candidate_FCM[n, i, j] = 0
                tmp = mf.Memebership(candidate_FCM[n, i, j])
                E[:, i, j] = E[:, i, j] + tmp  # Cumulative membership
    N = E[0] + E[1] + E[2]

    E_Res[0] = E[0]
    E_Res[1] = E[1]
    E_Res[2] = E[2]

    #Bayesian optimization
    a, r, Q, itr = bo.bayesian_optimization(N, E_Res)
    estimate_FCM = np.zeros(shape=(numRow, numCol))

    for i in range(numRow):
        for j in range(numCol):
            if Q[2, i, j] > 0.5:
                estimate_FCM[i, j] = 1  # Q[2] postive
            if Q[0, i, j] > 0.5:
                estimate_FCM[i, j] = -1  # Q[0] negative
    return estimate_FCM, Q