#include "ArducamDepthCamera.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void getPreview(uint8_t *preview_ptr, float *phase_image_ptr, float *amplitude_image_ptr)
{
    unsigned long int len = 240 * 180;
    for (unsigned long int i = 0; i < len; i++)
    {
        uint8_t amplitude = *(amplitude_image_ptr + i) > 30 ? 254 : 0;
        float phase = ((1 - (*(phase_image_ptr + i) / 2)) * 255);
        uint8_t depth = phase > 255 ? 255 : phase;
        *(preview_ptr + i) = depth & amplitude;
    }
}

int main()
{
    struct timespec start, end;
    double elapsed_time;
    ArducamDepthCamera tof = createArducamDepthCamera();
    ArducamFrameBuffer frame;
    if (arducamCameraOpen(tof,CSI,0))
        exit(-1);
    if (arducamCameraStart(tof,DEPTH_FRAME))
        exit(-1);
    uint8_t *preview_ptr = malloc(180*240*sizeof(uint8_t)) ;
    float* depth_ptr = 0;
    int16_t *raw_ptr = 0;
    float *amplitude_ptr = 0;
    ArducamFrameFormat format;
    if ((frame = arducamCameraRequestFrame(tof, 200)) != 0x00){
        format = arducamCameraGetFormat(frame,DEPTH_FRAME);
        arducamCameraReleaseFrame(tof,frame);  
    }
    printf("Starting timing\n");
    int n_frames = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (;;)
    {
        n_frames++;
        if ((frame = arducamCameraRequestFrame(tof, 200)) != 0x00)
        {
            depth_ptr = (float*)arducamCameraGetDepthData(frame);
            // printf("Center distance:%.2f.\n",depth_ptr[21600]);
            // amplitude_ptr = (float*)arducamCameraGetAmplitudeData(frame);
            // getPreview(preview_ptr,depth_ptr,amplitude_ptr);
            arducamCameraReleaseFrame(tof,frame);
        }
        if (n_frames == 1000)
            break;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("CPU time used: %f seconds\n", elapsed_time);
    printf("fps: %f \n", 1000 / elapsed_time);

    if (arducamCameraStop(tof))
        exit(-1);
    return 0;
}