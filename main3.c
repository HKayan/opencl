/**
 *
 *
 * Histogram equalization 
 *
 * Usage:  main <input.jpg> <output.jpg> 
 *
 * @author  hasan kayan
 *
 * @version 1.0, 30.01.2024
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

void load_image(const char* filepath, uint8_t** image, int* width, int* height) {
    int bpp;
    *image = stbi_load(filepath, width, height, &bpp, CHANNEL_NUM);
    if (*image == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }
}

void save_image(const char* filepath, const uint8_t* image, int width, int height) {
    if (!stbi_write_jpg(filepath, width, height, CHANNEL_NUM, image, 100)) {
        fprintf(stderr, "Error in saving the image\n");
        exit(1);
    }
}


void seq_histogram_equalizer(uint8_t* rgb_image,int width, int height);
void par_histogram_equalizer(uint8_t* rgb_image,int width, int height);

int main(int argc,char* argv[]) 
{		
    int width, height, bpp;
	
	// Reading the image in grey colors
    uint8_t* rgb_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);
	
    printf("Width: %d  Height: %d \n",width,height);
	printf("Input: %s , Output: %s  \n",argv[1],argv[2]);
	
	// start the timer
	double time1= omp_get_wtime();	
	
	seq_histogram_equalizer(rgb_image,width, height);
	//par_histogram_equalizer(rgb_image,width, height);
    
	double time2= omp_get_wtime();	
	printf("Elapsed time: %lf \n",time2-time1);	
	
	// Storing the image 
    stbi_write_jpg(argv[2], width, height, CHANNEL_NUM, rgb_image, 100);
    stbi_image_free(rgb_image);

    return 0;
}


void par_histogram_equalizer(uint8_t* rgb_image,int width, int height)
{

};

void seq_histogram_equalizer(uint8_t* rgb_image,int width, int height)
{	
	int *hist = (int*)calloc(256,sizeof(int));
	
	for(int i=0; i<height ; i++){
		for(int j=0; j<width; j++){
			hist[rgb_image[i*width + j]]++;
		}
	}
	
	double size = width * height;
   
     //cumulative sum for histogram values
    int cumhistogram[256];
	cumhistogram[0] = hist[0];
    for(int i = 1; i < 256; i++)
    {
        cumhistogram[i] = hist[i] + cumhistogram[i-1];
    }    
	
    int alpha[256];
    for(int i = 0; i < 256; i++)
    {
        alpha[i] = round((double)cumhistogram[i] * (255.0/size));
    }
			
    // histogram equlized image		
    for(int y = 0; y <height ; y++){
        for(int x = 0; x < width; x++){
            rgb_image[y*width + x] = alpha[rgb_image[y*width + x]];
		}
	}
	
}

