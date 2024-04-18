import numpy as np

fname = "faiss/siftsmall/siftsmall_base.fvecs"

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

print(fvecs_read(fname))