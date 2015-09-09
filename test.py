#!/usr/bin/env python

import pkg_resources
import pyopencl as cl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class convolve2d_OCL(object):

    def __init__(self):
        self.ctx = None
        self.fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    def __call__(self, ctx, src2, kernel):
        if self.ctx != ctx:
            self.ctx = ctx
            self.prg = cl.Program(self.ctx, pkg_resources.resource_string(__name__, "convolve2d.cl")).build()
        src = np.zeros((src2.shape[0],src2.shape[1],4),dtype=np.float32)
        src[:,:,0:src2.shape[2]] = src2[:,:,0:src2.shape[2]]
        kernel = np.array(kernel, dtype=np.float32)
        halflen = kernel.shape[0] / 2
        kernelf = kernel.flatten()
        shape = (src.shape[1], src.shape[0])
        src_buf = cl.image_from_array(self.ctx, src, 4)
        kernelf_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernelf)
        dest_buf = cl.Image(self.ctx, cl.mem_flags.WRITE_ONLY, self.fmt, shape=shape)
        queue = cl.CommandQueue(self.ctx)
        self.prg.BasicConvolve(queue, shape, None, src_buf, dest_buf, kernelf_buf, np.int_(kernelf.shape[0]))
        dest = np.empty_like(src)
        cl.enqueue_copy(queue, dest, dest_buf, origin=(0, 0), region=shape).wait()
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

src = mpimg.imread(pkg_resources.resource_filename(__name__, "PM5544_with_non-PAL_signals.png"))
#src = (src * 255).round().astype(np.uint8)
dest1 = convolve2d(ctx, src, kernel)
dest1 = (dest1 * 255).round().astype(np.uint8)
dest2 = convolve2d(ctx, src, kernel)

print "==============================="
print dest1
print "- - - - - - - - - - - - - - - -"
print dest2
print "==============================="
print np.array_equal(dest1,dest2)
print
print src.shape,src.dtype
print dest1.shape,dest1.dtype
print dest2.shape,dest2.dtype

fig = plt.figure(figsize=(20,10))
a=fig.add_subplot(1,3,1)
plt.imshow(src)
a.set_title("orig")
a=fig.add_subplot(1,3,2)
plt.imshow(dest1)
a.set_title("blurred 1")
a=fig.add_subplot(1,3,3)
plt.imshow(dest2)
a.set_title("blurred 2")
plt.show()
