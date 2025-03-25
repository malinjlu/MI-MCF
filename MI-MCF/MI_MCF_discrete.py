import numpy as np
import warnings
from itertools import combinations
from common.g2test import g2_test_dis
import copy
import matlab.engine
import matlab
from common.MI_discrete import MI_G2, mutual_information_test

eng = matlab.engine.start_matlab()

def MI_MCF_discrete(data, alpha, L,k1,k2):
    warnings.filterwarnings("ignore")
    n, p = np.shape(data)

    f = p - L # number of features
    f = int(f)
    p = int(p)

    PC = [[] for i in range(p)]
    da = data.tolist()
    for i in range(L):
        target = i + f
        pc, Result2, Result3 = MI_G2(data, target, alpha, p)

        if isinstance(pc, float):
            PC[target].append(int(pc))
        else:
            if len(pc) != 0:
                for i in pc:
                    PC[target].append(int(i))

    """get min dependency between PC feature and label"""
    min_dep = [1 for i in range(p)]
    for i in range(f,p):
        min_dep[i] = 1
        for j in PC[i]:
            pval,dep = g2_test_dis(data,i,j,[],alpha)
            if dep < min_dep[i]:
                min_dep[i] =dep

    """get label obtain other labels"""
    record = []
    for i in range(f, p):
        for j in range(f, p):
            if j in PC[i] and i not in record:
                record.append(i)
    record = sorted(list(set(record)))

    """get candidate features for every label"""
    fk_pc=[[] for i in range(p)]
    fea_lab = [[0 for i in range(f)] for i in range(p)]
    # print(np.shape(fea_lab))
    for j in record:
        pc_dep = []
        fk = [i for i in range(f) if i not in PC[j]]
        for k in fk:
            pval,dep = g2_test_dis(data,j,k,[],alpha)
            fea_lab[j][k] =dep
            if pval <= alpha and dep >= min_dep[j]:
               pc_dep.append([k,dep])
        pc_dep = sorted(pc_dep, key=lambda x: x[1], reverse=True) # sort features
        sellen = int(len(pc_dep) * k1) # select k1% features
        for k in range(sellen):
                fk_pc[j].append(pc_dep[k][0])

    pc_subset = [[] for i in range(p)]

    for j in record:
        for i in range(3):
            pc_subset_temp = subsets(PC[j], i)
            for k in pc_subset_temp:
                pc_subset[j].append(k)

    """phase 2"""
    for j in record:
        for i in range(f, p):  # label i
            if i in PC[j]:
                for fk in fk_pc[j]:
                    flag = False
                    for s in pc_subset[j]:
                        s_temp = s
                        if len(s_temp) < len(PC[j]) and i not in s_temp:
                            s_temp.append(i)
                        if i not in s_temp:
                            continue
                        pval,mi_xs = mutual_information_test(data,j,fk,s_temp,alpha)
                        if pval > alpha: # if independent
                            s_temp.remove(i)
                            vars = list(set(s_temp))
                            pval1, mi_xs= mutual_information_test(data,j,fk,vars,alpha)
                            if pval1 > alpha:
                                flag = True
                                break
                    if flag == False:
                        PC[j].append(fk)
                PC[j].remove(i)

    for i in range(f,p):
        PC[i]=list(set(PC[i]))
    for i in range(f,p):
        for j in range(f,p):
             if j in PC[i]:
                 PC[i].remove(j)

    """phase 3"""
    PC_temp = copy.deepcopy(PC)
    for i in range(f, p):
        pc_dep = []
        count = 0
        for j in PC_temp[i]:
            _, dep = g2_test_dis(data,j,i,[],alpha)
            pc_dep.append([j, dep])
        pc_dep = sorted(pc_dep, key=lambda x: x[1], reverse=False)  # sorted by dep from min to max
        sellen = int(len(pc_dep)*k2) # select k2% features
        for k in range(sellen):
            target = pc_dep[k][0]
            pc, Result2, Result3 = MI_G2(data, target, alpha, p)
            if isinstance(pc, float):
                if i != pc: # label i not in pc
                    PC[i].remove(target)
                else:
                    count = count + 1
            else:
                if i not in pc: # label i not in pc
                    PC[i].remove(target)
                else:
                    count = count + 1
            if count >= 1:
                break

    selfea = []
    for j in range(f, p):
        for i in PC[j]:
            if i not in selfea:
                selfea.append(i)

    selfea = list(set(selfea))
    return selfea

def subsets(nbrs,k):
    s=combinations(nbrs,k)
    sub=[]
    for i in s:
        sub.append(list(i))
    return sub