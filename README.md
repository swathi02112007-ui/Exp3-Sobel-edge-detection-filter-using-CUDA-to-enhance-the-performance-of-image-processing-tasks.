# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>NAME : SWATHI P N</h3>
<h3>REG NO : 212225230279</h3>
<h3>EX. NO. 03</h3>
<h3>DATE: 16-11-2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

```
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    if (x >= width || y >= height)
        return;

    int index = y * width + x;

    // Boundary pixel handling
    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1)
    {
        dstImage[index] = 0;
        return;
    }

    float Gx = 0.0f;
    float Gy = 0.0f;

    // Apply Sobel filters
    for (int ky = -1; ky <= 1; ky++)
    {
        for (int kx = -1; kx <= 1; kx++)
        {
            float pixel = srcImage[
                (y + ky) * width + (x + kx)
            ];

            Gx += pixel * Kx[ky + 1][kx + 1];
            Gy += pixel * Ky[ky + 1][kx + 1];
        }
    }

    // Calculate absolute gradients
    float Gx_abs = fabsf(Gx);
    float Gy_abs = fabsf(Gy);

    // Calculate gradient magnitude
    float magnitude = Gx_abs + Gy_abs;

    // Clamp output between 0 and 255
    if (magnitude > 255.0f)
        magnitude = 255.0f;

    dstImage[index] = (unsigned char)magnitude;
}

// CUDA error checking
void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n",
                cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Read input image
    Mat image = imread("/content/images.jpeg",
                       IMREAD_COLOR);

    if (image.empty())
    {
        printf("Error: Image not found.\n");
        return -1;
    }

    // Convert color image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    grayImage = grayImage.clone();

    int width = grayImage.cols;
    int height = grayImage.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == nullptr)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate GPU memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(cudaMalloc(
        &d_inputImage, imageSize));

    checkCudaErrors(cudaMalloc(
        &d_outputImage, imageSize));

    // Copy input image to GPU
    checkCudaErrors(cudaMemcpy(
        d_inputImage,
        grayImage.data,
        imageSize,
        cudaMemcpyHostToDevice));

    // Create CUDA events
    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Define block and grid dimensions
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // Launch CUDA kernel
    checkCudaErrors(cudaEventRecord(start));

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height
    );

    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate execution time
    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(
        &milliseconds, start, stop));

    // Copy output from GPU to CPU
    checkCudaErrors(cudaMemcpy(
        h_outputImage,
        d_outputImage,
        imageSize,
        cudaMemcpyDeviceToHost));

    // Create output image
    Mat outputImage(
        height,
        width,
        CV_8UC1,
        h_outputImage
    );

    // Save output image
    imwrite("output_sobel.jpeg", outputImage);

    // Display information
    printf("\nSobel Edge Detection Completed!\n");
    printf("Image Width: %d\n", width);
    printf("Image Height: %d\n", height);
    printf("Total time taken: %f milliseconds\n",
           milliseconds);

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## OUTPUT:

<img width="1118" height="751" alt="images" src="https://github.com/user-attachments/assets/b939e898-f71e-43b1-b173-4d1d5fc8720e" />

<img width="481" height="353" alt="image" src="https://github.com/user-attachments/assets/32e8db71-35e7-4ac3-bf03-c041ca34e935" />

## RESULT:
Thus the program has been executed by using CUDA that is Total time taken: 0.138816 milliseconds.

Questions:

1. What challenges did you face while implementing the Sobel filter for color images?
- Color images contain three channels: Red, Green and Blue.
- The Sobel filter requires grayscale pixel intensity values.
- Processing three channels increases memory usage and computation.
- To overcome this, the input color image was converted into grayscale using OpenCV.
- Boundary checks were added to avoid accessing pixels outside the image.
2. How did changing the block size influence the performance of your CUDA implementation?
- Block size determines the number of threads in each CUDA block.
- Smaller block sizes require more blocks to process the image.
- Larger block sizes can improve parallel execution but may increase resource usage.
- Different block sizes such as 8×8, 16×16 and 32×32 can produce different execution times.
- The best block size depends on the GPU and image dimensions.
3. What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
- Both CPU and CUDA implementations detect edges in an image.
- CUDA uses parallel processing, while CPU implementation generally uses fewer processing threads.
- Both outputs should show similar edge patterns.
- Minor differences may occur due to rounding, boundary handling and pixel intensity calculations.
- CUDA provides an opportunity for faster processing of large images.
4. Suggest potential optimizations for improving the performance of the Sobel filter.
- Use shared memory to reduce global memory access.
- Experiment with different CUDA block sizes.
- Use coalesced memory access.
- Minimize CPU-GPU data transfers.
- Store Sobel filter coefficients in constant memory.
- Optimize thread utilization.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

