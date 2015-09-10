#!/usr/bin/env python

import pkg_resources
from PIL import Image
import pyopencl as cl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class convolve2d_OCL(object):

    def __init__(self):
        self.ctx = None
    def __call__(self, ctx, src2, kernel):
        if self.ctx != ctx:
            self.ctx = ctx
            self.prg = cl.Program(self.ctx, pkg_resources.resource_string(__name__, "convolve2d.cl")).build()
        src2 = np.asarray(src2)#, dtype=np.uint8)
        src = np.zeros((src2.shape[0],src2.shape[1],4),dtype=src2.dtype)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        kernel = np.array(kernel, dtype=np.float32)
        kernelf = kernel.flatten()
        src_buf = cl.image_from_array(self.ctx, src, 4, norm_int=np.issubdtype(src.dtype, np.integer))
        kernelf_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernelf)
        dest_buf = cl.image_from_array(self.ctx, src.copy(), 4, mode="w",norm_int=np.issubdtype(src.dtype, np.integer))
        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve2d_naive(queue, (src.shape[1]-(kernelf.shape[0]>>1), src.shape[0]-(kernelf.shape[0]>>1)), None, src_buf, dest_buf, kernelf_buf, np.int_(kernelf.shape[0]))
        dest = np.empty_like(src)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()
        src_buf.release()
        dest_buf.release()
        kernelf_buf.release()
        return dest[:,:,0:src2.shape[2]].copy()
convolve2d = convolve2d_OCL()

ctx = cl.create_some_context()

kernel = [
    [1/16., 1/8., 1/16.],
    [1/8., 1/4., 1/8.],
    [1/16., 1/8., 1/16.],
]
print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
src1 = np.asarray(Image.open(pkg_resources.resource_filename(__name__, "PM5544_with_non-PAL_signals.png")))
dest1s1 = convolve2d(ctx, src1, kernel)
dest2s1 = convolve2d(ctx, src1, kernel)
print ("Equal:",np.array_equal(dest1s1,dest2s1))
print ("Dtype:",src1.dtype)
print ("Shape:",src1.shape)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
src2 = mpimg.imread(pkg_resources.resource_filename(__name__, "PM5544_with_non-PAL_signals.png"))
dest1s2 = convolve2d(ctx, src2, kernel)
dest2s2 = convolve2d(ctx, src2, kernel)
print ("Equal:",np.array_equal(dest1s2,dest2s2))
print ("Dtype:",src2.dtype)
print ("Shape:",src2.shape)

print dest1s1
print dest1s2

fig = plt.figure(figsize=(20,10))
a=fig.add_subplot(2,3,1)
plt.imshow(src1)
a.set_title("src1")
a=fig.add_subplot(2,3,2)
plt.imshow(dest1s1)
a.set_title("src1-blurred1")
a=fig.add_subplot(2,3,3)
plt.imshow(dest2s1)
a.set_title("src1-blurred2")
a=fig.add_subplot(2,3,4)
plt.imshow(src1)
a.set_title("src2")
a=fig.add_subplot(2,3,5)
plt.imshow(dest1s2)
a.set_title("src2-blurred1")
a=fig.add_subplot(2,3,6)
plt.imshow(dest2s2)
a.set_title("src2-blurred2")
plt.show()
