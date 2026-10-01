#pragma once
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

//frame headers
#define WS_FRAME_VIDEO 0x01
#define WS_FRAME_AUDIO_TX 0x02

//struct for buffer
typedef struct {
    uint8_t *data;
    size_t len;
} video_frame_t;

typedef struct {
    int16_t samples[512];
    size_t count;
} audio_buffer_t;

// global queues
extern QueueHandle_t video_queue;
extern QueueHandle_t audio_tx_queue;

//task handles
extern TaskHandle_t camera_task_handle;
extern TaskHandle_t audio_tx_task_handle;
extern TaskHandle_t ws_send_task_handle;