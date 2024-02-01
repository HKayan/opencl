#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define HISTOGRAM_SIZE 256

// Function prototypes
void load_image(const char* filepath, uint8_t** image, int* width, int* height, int* channels);
void save_image(const char* filepath, const uint8_t* image, int width, int height, int channels);
const char* LoadKernelSource() {
    const char* kernelSource = 
        "__kernel void calculate_histogram(__global const uchar *image, __global int *hist, const int numPixels) {\n"
    "    int id = get_global_id(0);\n"
    "    if (id < numPixels) {\n"
    "        atomic_inc(&hist[image[id]]);\n"
    "    }\n"
    "}\n"

    "__kernel void cumulative_histogram(__global const int *hist, __global int *cumHist, const int histSize, __local int *temp) {\n"
    "    int gid = get_global_id(0);\n"
    "    int lid = get_local_id(0);\n"
    "\n"
    "    // Load histogram into local memory\n"
    "    if (lid < histSize) {\n"
    "        temp[lid] = hist[lid];\n"
    "    } else {\n"
    "        temp[lid] = 0;\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "\n"
    "    // Perform parallel scan (prefix sum) in local memory\n"
    "    for (int stride = 1; stride < get_local_size(0); stride *= 2) {\n"
    "        int index = (lid + 1) * stride * 2 - 1;\n"
    "        if (index < get_local_size(0)) {\n"
    "            temp[index] += temp[index - stride];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "\n"
    "    // Write back to global memory\n"
    "    if (lid < histSize) {\n"
    "        cumHist[lid] = temp[lid];\n"
    "    }\n"
    "}\n"

    "__kernel void apply_equalization(\n"
    "    __global uchar *image, \n"
    "    __global const int *cumHist, \n"
    "    const int totalPixels, \n"
    "    const float scale_factor)\n"
    "{\n"
    "    int id = get_global_id(0);\n"
    "    if (id < totalPixels) {\n"
    "        // Retrieve the original pixel value\n"
    "        uchar pixelValue = image[id];\n"
    "        \n"
    "        // Calculate the new pixel value based on the cumulative histogram\n"
    "        // Normalize the cumulative histogram value to the range [0, 255]\n"
    "        int newValue = (int)(255.0f * (cumHist[pixelValue] - cumHist[0]) * scale_factor);\n"
    "        \n"
    "        // Clamp the result to avoid overflow\n"
    "        newValue = newValue > 255 ? 255 : newValue < 0 ? 0 : newValue;\n"
    "        \n"
    "        // Update the image with the new pixel value\n"
    "        image[id] = (uchar)newValue;\n"
    "    }\n"
    "}\n";

    return kernelSource;
}


int main() {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel histogram_kernel = NULL, cumhist_kernel = NULL, equalization_kernel = NULL;
    cl_mem image_buffer = NULL, hist_buffer = NULL, cumhist_buffer = NULL;

    int width, height, channels;
    uint8_t* input_image;

    // Load the image
    load_image("papagan3.jpg", &input_image, &width, &height, &channels);
    if (input_image == NULL) {
        fprintf(stderr, "Failed to load the image.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform ID. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get device ID. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL context. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
    const cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Create and build the program
    const char* kernelSource = LoadKernelSource();
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create CL program. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Check for build errors
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Failed to build CL program. Error: %d\nLog:\n%s\n", err, log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // Create kernels
    histogram_kernel = clCreateKernel(program, "calculate_histogram", &err);
    if (!histogram_kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create calculate_histogram kernel. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
    cumhist_kernel = clCreateKernel(program, "cumulative_histogram", &err);
    equalization_kernel = clCreateKernel(program, "apply_equalization", &err);

    // Create buffers
    int total_pixels = width * height;
    image_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, total_pixels * sizeof(uint8_t), NULL, &err);
    hist_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, HISTOGRAM_SIZE * sizeof(int), NULL, &err);
    cumhist_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, HISTOGRAM_SIZE * sizeof(int), NULL, &err);

    // Set kernel arguments and enqueue kernels for execution (omitted for brevity)
    err = clSetKernelArg(histogram_kernel, 0, sizeof(cl_mem), &image_buffer);
    err |= clSetKernelArg(histogram_kernel, 1, sizeof(cl_mem), &hist_buffer);
    err |= clSetKernelArg(histogram_kernel, 2, sizeof(int), &total_pixels);

    // Enqueue calculate_histogram kernel
    size_t global_work_size_histogram[1] = {total_pixels};
    err = clEnqueueNDRangeKernel(queue, histogram_kernel, 1, NULL, global_work_size_histogram, NULL, 0, NULL, NULL);
    

    int histogram_size = HISTOGRAM_SIZE;
    // Set kernel arguments for cumhist_kernel
    err = clSetKernelArg(cumhist_kernel, 0, sizeof(cl_mem), &hist_buffer);
    err |= clSetKernelArg(cumhist_kernel, 1, sizeof(cl_mem), &cumhist_buffer);
    err |= clSetKernelArg(cumhist_kernel, 2, sizeof(int), &histogram_size);
    size_t local_memory_size = HISTOGRAM_SIZE * sizeof(int);
    err |= clSetKernelArg(cumhist_kernel, 3, local_memory_size, NULL); // Local memory

    // Enqueue cumhist_kernel
    size_t global_work_size_cumhist[1] = {HISTOGRAM_SIZE}; // One work-item per histogram bin
    size_t local_work_size_cumhist[1] = {HISTOGRAM_SIZE}; // Adjust based on your device's capabilities
    err = clEnqueueNDRangeKernel(queue, cumhist_kernel, 1, NULL, global_work_size_cumhist, local_work_size_cumhist, 0, NULL, NULL);

    // Set kernel arguments for equalization_kernel
    err = clSetKernelArg(equalization_kernel, 0, sizeof(cl_mem), &image_buffer);
    err |= clSetKernelArg(equalization_kernel, 1, sizeof(cl_mem), &cumhist_buffer);
    err |= clSetKernelArg(equalization_kernel, 2, sizeof(int), &total_pixels);
    float scale_factor = 1.0f / total_pixels; // Assuming equalization logic requires this
    err |= clSetKernelArg(equalization_kernel, 3, sizeof(float), &scale_factor);

    // Enqueue equalization_kernel
    size_t global_work_size_equalization[1] = {total_pixels}; // One work-item per pixel
    err = clEnqueueNDRangeKernel(queue, equalization_kernel, 1, NULL, global_work_size_equalization, NULL, 0, NULL, NULL);


    // Read back the result and cleanup
    err = clEnqueueReadBuffer(queue, image_buffer, CL_TRUE, 0, total_pixels * sizeof(uint8_t), input_image, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to read buffer. Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
    save_image("output.jpg", input_image, width, height, channels);
    stbi_image_free(input_image);

    // Release resources
    clReleaseMemObject(image_buffer);
    clReleaseMemObject(hist_buffer);
    clReleaseMemObject(cumhist_buffer);
    clReleaseKernel(histogram_kernel);
    clReleaseKernel(cumhist_kernel);
    clReleaseKernel(equalization_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}

void load_image(const char* filepath, uint8_t** image, int* width, int* height, int* channels) {
    *image = stbi_load(filepath, width, height, channels, 0);
    if (*image == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }
}

void save_image(const char* filepath, const uint8_t* image, int width, int height, int channels) {
    if (!stbi_write_jpg(filepath, width, height, channels, image, 100)) {
        fprintf(stderr, "Error in saving the image\n");
        exit(1);
    }
}
