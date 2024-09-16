#-*- coding = utf-8 -*-
#@Time : 2024/3/28 19:15
#@File : NRSEL_SFP.py
#@Software : PyCharm
import copy
import random, math
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.neighbors import KDTree
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef
# Calculate the neighborhood class of each sample with respect to the given attribute subset

def APartition(data):
    KDT = KDTree(data, metric='euclidean')                            # Call the kd-tree algorithm
    Equivalence = KDT.query_radius(data,r=delta)        #  Calculate the number of samples whose distance from the current sample is less than delta
    return Equivalence

# Calculate the neighborhood class of each sample with respect to the decision attribute
def DPartition(data):
    Equivalence = []
    for num in range(max(data)+1):
        Equivalence.append([i for (i,j) in enumerate(data) if j==num])
    return Equivalence

# Calculate the relative positive region
def RelevantPositive(data, D_equval):
    temp_under_C_D = []
    temp_equval = APartition(data)
    index_count = 0
    for every_neighbor in temp_equval:
        temp = every_neighbor.tolist()
        if set(temp) < set(D_equval[0]):
            temp_under_C_D.append(index_count)
        elif set(temp) < set(D_equval[1]):
            temp_under_C_D.append(index_count)
        index_count += 1
    return temp_under_C_D
if __name__ == '__main__':
   # dataSet = pd.read_csv("filepath")       # Use the pandas.read_csv function to read a CSV formatted dataset, where filepath is the file path

  #  dataSet = pd.read_csv("filepath")
    dataSet = pd.read_csv(r'D:\workspace\python\data\defect prediction\KC3.csv')
    ORI_X = dataSet.iloc[:,:-1]             # Obtain the columns of condition attributes of the dataset
    ORI_y = dataSet.iloc[:,-1]              # Obtain the column of decision attribute of the dataset

    # Set the ensemble size M, the neighborhood radius delta, and the parameter ε
    # M = a value in the interval [5,50]                        
    # delta = a value in the interval [0.01, 0.3]                  
    # epsilon = a value in the interval [0.75,0.99]

    # Convert the character based decision attribute values to discrete values
    class_label = LabelEncoder()
    ORI_y = class_label.fit_transform(ORI_y)

    # Use the SMOTE algorithm to handle the problem of imbalanced data, where the parameter sampling_strategy is set to 0.8
    smo = SMOTE(sampling_strategy=0.8)
    X, y = smo.fit_resample(ORI_X, ORI_y)

    # data normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    D_Equivalence = DPartition(y)                   # Calculate the neighborhood class of the sample on D

    POS_C_D = RelevantPositive(X, D_Equivalence)    # Calculate the relative positive region of D with respect to C


    C = [i for i in range(X.shape[1])]
    Set_AR = []              
            
    num_column = dataSet.columns.size
    random_n = random.randint(math.ceil((num_column-1)*0.2), math.ceil((num_column-1)*0.5))  # generate a random integer in the interval [⌈|C|*0.2⌉, ⌈|C|*0.5⌉]
    print(random_n)
    attr_set = range(num_column-1)    # Use the list attr_set to represent the set of condition attributes                          
    temp_s = Series(attr_set)    # Convert the list attr_set into a Series object, in order to conduct the following sampling operation
    random_feature = temp_s.sample(random_n, replace=False) # random_feature stores random_n features obtained from C by random sampling without replacement      
    print(random_feature)
    print(set(random_feature))
    std_Rem = list(set(C) - set(random_feature))
    iter_t = 0

    # Calculate the set of NARs
    while len(Set_AR) < M:                                  # When the number of NARs is less than M, repeat the following operation
        AR = copy.deepcopy(list(random_feature))                        # Let AR=random_feature represent the current NAR
        Rem = std_Rem     
        AR_X = X[:,AR]                              
        POS_AR_D = RelevantPositive(AR_X, D_Equivalence)    # Compute the positive region of D with respect to AR
        while (len(POS_C_D)-len(POS_AR_D))> ((1- epsilon) * (len(POS_C_D)+1)):       # Check whether AR meets the conditions of neighborhood ε-approximate reduct
            arry = np.zeros((len(Rem),2),dtype='object')
            tempCount = 0
            print(Rem)
            for c in Rem:
                tempXc = X[:, c]
                tempXc = tempXc.reshape(-1,1)
                c_AR_X = np.concatenate((AR_X, tempXc), axis=1)
                POS_c_AR_D = RelevantPositive(c_AR_X, D_Equivalence)
                SGF_c_AR_D = len(POS_c_AR_D) - len(POS_AR_D)
                arry[tempCount][0] = c
                arry[tempCount][1] = SGF_c_AR_D
                tempCount += 1
            arry = arry[(list(np.argsort(-arry, 0)[:, -1])), :]
            tempMax = arry[0][0]
            AR.append(tempMax)
            Rem.remove(tempMax)
            AR_X = X[:,AR]
            POS_AR_D = RelevantPositive(AR_X, D_Equivalence)
            
        if AR not in Set_AR:
            iter_t += 1
            print("generating the",iter_t,"-th NAR")
            Set_AR.append(copy.deepcopy(AR))
    print(Set_AR)


    # Choose the classification algorithm (KNN or CART)
    clf = KNeighborsClassifier(n_neighbors=5)
    # clf = DecisionTreeClassifier(criterion='gini')
    classification_result = []
    Y_test = []

    # Ensemble Learning --- Building a Base Learner BL
    for reduction in Set_AR:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        temp_s = Series(x_train)    # Convert x_train into a Series object, in order to conduct the following sampling operation
        sample_train = temp_s.sample(frac=1, replace=True) # generate the jth bootstrap sample by resampling with replacement        
        reduct_x_train = sample_train[:,reduction]
        reduct_x_test = x_test[:,reduction]
        clf.fit(reduct_x_train,y_train)
        Y_test.append(copy.deepcopy(list(y_test)))
        classification_result.append(copy.deepcopy(list(clf.predict(reduct_x_test))))
    result = np.array(classification_result,dtype='int32')

    y_predict = []
    count_test = len(x_test)                # Count_test represents the number of test samples
    count_correct = 0                       # Count_comrrect represents the current number of correctly predicted test samples by the ensemble learner
    for i in range(count_test):             # For the i-th test sample, obtain the classification result of the ensemble learner for the test sample through voting, i.e. np.argmax(counts)

        counts = np.bincount(result[:, i])
        y_predict.append(copy.deepcopy(np.argmax(counts)))
        for j in range(M):
            if np.argmax(counts) == Y_test[j][i]:
                count_correct = count_correct + 1
    precision = count_correct / (float(count_test))/M

    temp_recall = 0.0
    temp_precision = 0.0
    temp_f1score = 0.0
    temp_auc = 0.0
    temp_accuracy = 0.0
    temp_mcc = 0.0
    for i in range(M):
        temp_recall = temp_recall + recall_score(Y_test[i], y_predict, average='macro')
        temp_precision = temp_precision + precision_score(Y_test[i], y_predict, average='macro')
        temp_f1score = temp_f1score + f1_score(Y_test[i], y_predict, average='macro')
        temp_auc = temp_auc + roc_auc_score(Y_test[i], y_predict, average='macro')
        temp_accuracy = temp_accuracy + accuracy_score(Y_test[i], y_predict)
        temp_mcc = temp_mcc + matthews_corrcoef(Y_test[i], y_predict)
    recall = temp_recall / M
    tolprecision = temp_precision / M
    f1score = temp_f1score / M
    AUC = temp_auc / M
    accuracy = temp_accuracy / M
    mcc = temp_mcc / M
    print(recall)
    print(AUC)
    print(f1score)
    print(mcc)












