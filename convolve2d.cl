__kernel void convolve2d_naive(__read_only  image2d_t imgSrc,
                               __write_only image2d_t imgConvolved,
                               __constant   float *   kernelValues,
                                            int       w)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

   for (int i = 0; i < w; i++)
   {
       for (int j = 0; j < w; j++)
       {
          convPix += read_imagef(imgSrc, (int2)(x+i,y+j)) * kernelValues[i + w*j];
       }
   }
   write_imagef(imgConvolved, (int2)(x + (w>>1), y + (w>>1)), convPix);
}

#define BLOCK_SIZE 16

__kernel void convolve2d_local(__read_only  image2d_t imgSrc,
                               __write_only image2d_t imgConvolved,
                               __constant   float *   kernelValues,
                                            int       w)
{
   const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
         CLK_ADDRESS_CLAMP | //Clamp to zeros
         CLK_FILTER_NEAREST; //Don't interpolate

   int wBy2 = w>>1; //w divided by 2

   //Goes up to 21x21 filters
    __local uint4 P[BLOCK_SIZE+20][BLOCK_SIZE+20];

    //Identification of this workgroup
   int i = get_group_id(0);
   int j = get_group_id(1);

   //Identification of work-item
   int idX = get_local_id(0);
   int idY = get_local_id(1);

   int ii = i*BLOCK_SIZE + idX; // == get_global_id(0);
   int jj = j*BLOCK_SIZE + idY; // == get_global_id(1);

   int2 coords = (int2)(ii, jj);

   //Reads pixels
   P[idX][idY] = read_imageui(imgSrc, smp, coords);

   //Needs to read extra elements for the filter in the borders
   if (idX < w)
   {
      coords.x = ii + BLOCK_SIZE; coords.y = jj;
      P[idX + BLOCK_SIZE][idY] = read_imageui(imgSrc, smp, coords);
   }

   if (idY < w)
   {
      coords.x = ii; coords.y = jj + BLOCK_SIZE;
      P[idX][idY + BLOCK_SIZE] = read_imageui(imgSrc, smp, coords);
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   //Computes convolution
   float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
   float4 temp;

   for (int ix = 0; ix < w; ix++)
   {
       int tx = idX + ix;
       for (int jy = 0; jy < w; jy++)
       {
          int ty = idY + jy;
          temp = (float4)((float)P[tx][ty].x, (float)P[tx][ty].y, (float)P[tx][ty].z, (float)P[tx][ty].w);
          convPix += temp * kernelValues[ix + w*jy];
       }
   }

   P[idX+wBy2][idY+wBy2] = (uint4)((uint)convPix.x, (uint)convPix.y, (uint)convPix.z, (uint)convPix.w);

   barrier(CLK_LOCAL_MEM_FENCE);
   coords = (int2)(ii+wBy2, jj+wBy2);
   write_imageui(imgConvolved, coords, P[idX+wBy2][idY+wBy2]);
}
