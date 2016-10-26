import numpy as np 


pos = np.load("posproc.npy")
neg = np.load("negproc.npy")

lpos = len(pos)
lneg = len(neg)

lpos_train = lpos * 0.8
lneg_train = lneg * 0.8

pos_train = pos[:lpos_train]
neg_train = neg[:lneg_train]

pos_test = pos[lpos_train:]
neg_test = neg[lneg_train:]

np.save("postrain", pos_train)
np.save("negtrain", neg_train)

np.save("postest", pos_test)
np.save("negtest", neg_test)

