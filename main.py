import scipy.io as sio
import numpy as np
import Estimate_FCM as ef
import Cal_Index_print as cip
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import os



if __name__ == '__main__':

    Case = 4 # 1-4

    if Case == 1:
        # parameters
        NN = 'node10' # node5
        #
        path = 'data/Case1/'+NN+'/'
        # read real FCM
        TrueNN = np.genfromtxt(path + NN +'_TrueNN'+ '.txt', delimiter=',')

        # read candidate_FCMs
        candidate_FCM = []
        for i in range(10):
            tmp = np.genfromtxt(path + NN + '_' + str(i + 1) + '.txt', delimiter=',')
            candidate_FCM.append(tmp)
        candidate_FCM = np.array(candidate_FCM)

    elif Case == 2:
        # parameters
        data = 'synthetic_3' # synthetic_1: noise and missing;   synthetic_2 and synthetic_3:only noise
        method = 'DE'  # L1,L2,OnlineFCM,-ACOR,RCCGA,DE
        numFCM = 500 # The number of candidate FCMs
        noisy = 0.001
        missing = 0
        #

        candidate_FCM = []
        filename = []

        path = 'data/Case2/' + method + '/' + data + '/'
        if method=='L1' or method=='L2' or method =='OnlineFCM':
            if missing !=0 and data != 'synthetic_1':
                print('ERROR: Only synthetic_1 contain missing value!\n')
                os._exit()
            filename = path + data+'_noisy_' + str(noisy) + '_missing_' + str(missing) + '_BestNN_'

        elif method=='ACOR' or method=='DE' or method =='RCGA':
            filename = path + data + '_' + method + '_BestNN_noisy_' + str(noisy) + '_missing_' + str(missing) +'_'

        # read real FCM
        if data == 'synthetic_1':
            TrueNN_path = path + '/synthetic_1_n5_d40_s40_t10_TrueNN.mat'
        elif data == 'synthetic_2':
            TrueNN_path = path + '/synthetic_2_n10_d40_s40_t10_TrueNN.mat'
        elif data == 'synthetic_3':
            TrueNN_path = path + '/synthetic_3_n20_d40_s40_t10_TrueNN.mat'
        data_tmp = sio.loadmat(TrueNN_path)
        TrueNN = data_tmp['TrueNN']

        # read candidate_FCMs
        for i in range(numFCM):
            data_tmp = sio.loadmat(filename+str(i+1)+'.mat')
            data = data_tmp['BestNN']
            candidate_FCM.append(data)
        candidate_FCM = np.array(candidate_FCM)

    elif Case == 3:
        # parameters
        numFCM = 10  # Each method takes the first 10 FCM of the result
        noisy = 0.001
        data = 'synthetic_2'  # synthetic_2 and synthetic_3:only noise

        # read real FCM
        root_path = 'data/Case3/'
        path = 'data/Case3/' + 'L1' + '/' + data + '/'
        if data == 'synthetic_2':
            TrueNN_path = path + '/synthetic_2_n10_d40_s40_t10_TrueNN.mat'
        elif data == 'synthetic_3':
            TrueNN_path = path + '/synthetic_3_n20_d40_s40_t10_TrueNN.mat'
        data_tmp = sio.loadmat(TrueNN_path)
        TrueNN = data_tmp['TrueNN']

        # read candidate_FCMs
        candidate_FCM = []
        # Lasso-FCM
        L1_filename = root_path + 'L1' + '/' + data + '/' + data + '_noisy_' + str(noisy) + '_missing_' + str(0) + '_BestNN_'
        for i in range(numFCM):
            L1_data_tmp = sio.loadmat(L1_filename + str(i + 1) + '.mat')
            L1_data = L1_data_tmp['BestNN']
            candidate_FCM.append(L1_data)
        # Ridge-FCM
        L2_filename = root_path + 'L2' + '/' + data + '/' + data + '_noisy_' + str(noisy) + '_missing_' + str(0) + '_BestNN_'
        for i in range(numFCM):
            L2_data_tmp = sio.loadmat(L2_filename + str(i + 1) + '.mat')
            L2_data = L2_data_tmp['BestNN']
            candidate_FCM.append(L2_data)
        # Online-FCM
        Online_filename = root_path + 'OnlineFCM' + '/' + data + '/' + data + '_noisy_' + str(noisy) + '_missing_' + str(0) + '_BestNN_'
        for i in range(numFCM):
            Online_data_tmp = sio.loadmat(Online_filename + str(i + 1) + '.mat')
            Online_data = Online_data_tmp['BestNN']
            candidate_FCM.append(Online_data)
        # ACORD
        ACOR_filename = root_path + 'ACOR' + '/' + data + '/' + data + '_' + 'ACOR' + '_BestNN_noisy_' + str(noisy) + '_missing_' + str(0) + '_'
        for i in range(numFCM):
            ACOR_data_tmp = sio.loadmat(ACOR_filename + str(i + 1) + '.mat')
            ACOR_data = ACOR_data_tmp['BestNN']
            candidate_FCM.append(ACOR_data)
        # RCGA
        RCGA_filename = root_path + 'RCGA' + '/' + data + '/' + data + '_' + 'RCGA' + '_BestNN_noisy_' + str(noisy) + '_missing_' + str(0) + '_'
        for i in range(numFCM):
            RCGA_data_tmp = sio.loadmat(RCGA_filename + str(i + 1) + '.mat')
            RCGA_data = RCGA_data_tmp['BestNN']
            candidate_FCM.append(RCGA_data)
        # DE
        DE_filename = root_path + 'DE' + '/' + data + '/' + data + '_' + 'DE' + '_BestNN_noisy_' + str(noisy) + '_missing_' + str(0) + '_'
        for i in range(numFCM):
            DE_data_tmp = sio.loadmat(DE_filename + str(i + 1) + '.mat')
            DE_data = DE_data_tmp['BestNN']
            candidate_FCM.append(DE_data)

        candidate_FCM = np.array(candidate_FCM)

    elif Case == 4:
        # parameters
        # method = 'L1_i'  #L1_i, OnlineFCM_delta
        method = 'OnlineFCM_delta'  #L1_i, OnlineFCM_delta
        dataname = 'Dream10'
        # parameters

        filename = 'data/Case4/' + method +'/' + dataname + '/' + dataname +'_NN_lamda_'
        candidate_FCM = []

        if method == 'L1_i':
            # parameters
            numFCM = 50 # 10,20,30,40,50
            #
            for i in range(numFCM):
                # 循环读取文件
                data_tmp = sio.loadmat(filename + str(i + 1) + '.mat')
                data = data_tmp['BestNN']
                candidate_FCM.append(data)

        elif method == 'OnlineFCM_delta':
            # parameters
            delta = 5 #2,3,4,5,6
            #
            # read candidate_FCMs
            numFCM = delta ** 3
            index_list = []
            index_arr = np.zeros(shape=(6, 6, 6))
            index = 1
            for l1 in range(6):
                for l2 in range(6):
                    for a1 in range(6):
                        index_arr[l1, l2, a1] = index
                        index = index + 1

            if delta == 6:
                for i in range(numFCM):
                    # 循环读取文件
                    data_tmp = sio.loadmat(filename + str(i + 1) + '.mat')
                    data = data_tmp['BestNN']
                    candidate_FCM.append(data)
            elif delta == 5:
                index_list = index_arr[1:, :-1, 1:].flatten().tolist()
            elif delta == 4:
                index_list = index_arr[2:, :-2, 2:].flatten().tolist()
            elif delta == 3:
                index_list = index_arr[3:, :-3, 3:].flatten().tolist()
            elif delta == 2:
                index_list = index_arr[4:, :-4, 4:].flatten().tolist()

            for i in index_list:
                data_tmp = sio.loadmat(filename + str(int(i)) + '.mat')
                data = data_tmp['BestNN']
                candidate_FCM.append(data)
        candidate_FCM = np.array(candidate_FCM)

        # read real FCM
        path = 'data/Case4/'+ method +'/'+dataname
        if dataname == 'Dream10':
            TrueNN_path = path + '/Dream_node_10_s40_t21signed_TrueNN.mat'
        elif dataname == 'Dream100':
            TrueNN_path = path + '/Dream_node_100_s100_t21signed_TrueNN.mat'
        data_tmp = sio.loadmat(TrueNN_path)
        TrueNN = data_tmp['TrueNN']




    #estimate FCM from candidate FCMs
    estimate_FCM, K = ef.cal_estimate_FCM(candidate_FCM)

    #Calulate SS_Mean and AUC
    cip.cal_index_print(TrueNN, estimate_FCM, candidate_FCM, K)
