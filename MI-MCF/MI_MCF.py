from MI_MCF_discrete import MI_MCF_discrete
import matlab.engine
import matlab
import scipy.io as sio
eng = matlab.engine.start_matlab()

def MI_MCF(data, alpha, L, k1, k2,datatype):
    datatype == 'D'
    selfea = MI_MCF_discrete(data,alpha,L,k1,k2)

    return selfea
if __name__ == "__main__":
    """
    An example of MI-MCF
    """
    dataset = sio.loadmat("data\Flags-train.mat")
    data = dataset["train"]
    alpha = 0.05 # Significance level
    L = 7 # number of labels
    k1 = 0.7;k2 = 0.1 # parameters
    datatype = "D" # discrete or continuous
    selfea = MI_MCF(data,alpha,L,k1,k2,datatype)
    print(selfea)
