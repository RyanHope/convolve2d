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
        src2 = np.asarray(src2)
        src = np.zeros((src2.shape[0], src2.shape[1], 4),dtype=src2.dtype)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        norm = np.issubdtype(src.dtype, np.integer)
        src_buf = cl.image_from_array(self.ctx, src, 4, norm_int=norm)
        dest_buf = cl.image_from_array(self.ctx, src, 4, mode="w", norm_int=norm)
        dest = np.empty_like(src)
        kernel = np.array(kernel, dtype=np.float32)
        kernelf = kernel.flatten()
        kernelf_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernelf)
        halflen = (kernelf.shape[0]>>1)
        queue = cl.CommandQueue(self.ctx)
        self.prg.convolve2d_naive(queue, (src.shape[1]-halflen, src.shape[0]-halflen), None, src_buf, dest_buf, kernelf_buf, np.int_(kernelf.shape[0]))
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=(src.shape[1], src.shape[0])).wait()
        dest = dest[:,:,0:src2.shape[2]].copy()
        src_buf.release()
        dest_buf.release()
        kernelf_buf.release()
        return dest
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
print ("Convolutions Equal:",np.array_equal(dest1s1,dest2s1))
print ("Source Dtype:",src1.dtype)
print ("Source Shape:",src1.shape)
print ("Convolve1 Dtype:",dest1s1.dtype)
print ("Convolve1 Shape:",dest1s1.shape)
print ("Convolve2 Dtype:",dest2s1.dtype)
print ("Convolve2 Shape:",dest2s1.shape)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
src2 = mpimg.imread(pkg_resources.resource_filename(__name__, "PM5544_with_non-PAL_signals.png"))
dest1s2 = convolve2d(ctx, src2, kernel)
dest2s2 = convolve2d(ctx, src2, kernel)
print ("Convolutions Equal:",np.array_equal(dest1s2,dest2s2))
print ("Source Dtype:",src2.dtype)
print ("Source Shape:",src2.shape)
print ("Convolve1 Dtype:",dest1s2.dtype)
print ("Convolve1 Shape:",dest1s2.shape)
print ("Convolve2 Dtype:",dest2s2.dtype)
print ("Convolve2 Shape:",dest2s2.shape)

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
