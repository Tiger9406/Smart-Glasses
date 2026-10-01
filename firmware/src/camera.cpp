#include "camera.h"
#include "config.h"
#include "esp_camera.h"
#include "camera_pins.h"
#include "led.h"

void setupCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    if(psramFound()){
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        config.frame_size = FRAMESIZE_SVGA;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }
    
    esp_camera_init(&config);
    sensor_t * s = esp_camera_sensor_get();
    // s->set_vflip(s, 1);
}

// define camera task of getting data and sending to global queue
void camera_task(void *pvParameters){
    camera_fb_t *fb = NULL;
    video_frame_t frame;

    setLED(true);
    // static uint32_t video_drops=0;
    
    while (1) {
        fb = esp_camera_fb_get();
        if (fb) {
            if (fb->format == PIXFORMAT_JPEG) { //image quality defined here
                frame.data = (uint8_t*)malloc(fb->len);
                if (frame.data) {
                    memcpy(frame.data, fb->buf, fb->len);
                    frame.len = fb->len;

                    // audio: want to keep as much as possible
                    // but pictures, chuck old ones out
                    if (xQueueSend(video_queue, &frame, 0) != pdTRUE) {
                        video_frame_t old;
                        if (xQueueReceive(video_queue, &old, 0) == pdTRUE)
                            free(old.data);
                        if (xQueueSend(video_queue, &frame, 0) != pdTRUE)
                            free(frame.data);
                    }
                    
                    /*
                    if (xQueueSend(video_queue, &frame, 0) != pdTRUE) {
                        free(frame.data);
                        // video_drops++;
                        // Serial.printf("[CAMERA] Queue full! Dropped frame. Total drops: %u\n", video_drops);
                    }
                    */
                }
            }
            esp_camera_fb_return(fb);
        }
        vTaskDelay(pdMS_TO_TICKS(100)); //roughly 10 fps; could improve
    }
}
