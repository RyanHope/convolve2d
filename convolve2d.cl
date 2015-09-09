#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable

__kernel void BasicConvolve(__read_only  image2d_t imgSrc,
                            __write_only image2d_t imgConvolved,
                            __global     float *   kernelValues,
                            __global     int *     kernelSize)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   //Kernel size (ideally, odd number)
   //global_size should be [width-w/2, height-w/2]
   //Writes answer to [x+w/2, y+w/2]
   int w = kernelSize[0];

   int x = get_global_id(0);
   int y = get_global_id(1);

   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
   float4 temp;
   uint4 pix;

   int2 coords;

   for (int i = 0; i < w; i++)
   {
       for (int j = 0; j < w; j++)
       {
          coords.x = x+i; coords.y = y+j;

          pix = read_imageui(imgSrc, smp, coords);
          temp = (float4)((float)pix.x, (float)pix.y, (float)pix.z, (float)pix.w);

          convPix += temp * kernelValues[i + w*j];
       }
   }

   coords.x = x + (w>>1); coords.y = y + (w>>1);
   pix = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);
   write_imageui(imgConvolved, coords, pix);
}
