__kernel void BasicConvolve(__read_only  image2d_t imgSrc,
                            __write_only image2d_t imgConvolved,
                            __constant   float *   kernelValues,
                                         int       w)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
   int x = get_global_id(0);
   int y = get_global_id(1);
   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
   for (int i = 0; i < w; i++)
   {
       for (int j = 0; j < w; j++)
       {
          convPix += read_imagef(imgSrc, smp, (int2)(x+i,y+j)) * kernelValues[i + w*j];
       }
   }
   write_imagef(imgConvolved, (int2)(x + (w>>1), y + (w>>1)), convPix);
}
