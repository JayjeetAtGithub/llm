import numpy as np
import struct

def fvecs_read(filename, c_contiguous=True):
    with open(filename, 'rb') as f:
        x = f.read(4)
        int_val = int.from_bytes(x, "little")
        print(int_val)

        a = f.read(4)
        val = struct.unpack('f', a)
        print(val)

        b = f.read(4)
        val = struct.unpack('f', b)
        print(val)
    # print(fv)
    # if fv.size == 0:
    #     return np.zeros((0, 0))
    # dim = fv.view(np.int32)[0]
    # assert dim > 0
    # fv = fv.reshape(-1, 1 + dim)
    # if not all(fv.view(np.int32)[:, 0] == dim):
    #     raise IOError("Non-uniform vector sizes in " + filename)
    # fv = fv[:, 1:]
    # if c_contiguous:
    #     fv = fv.copy()
    # return fv

out = fvecs_read('faiss/siftsmall/siftsmall_base.fvecs')
print(out)
