#include <Arduino.h>
#include "config.h"
#include "camera.h"
#include "audio.h"
// #include "network.h"
#include "udp.h"
#include "led.h"

#include "secrets.h"

// Define Globals
QueueHandle_t video_queue = NULL;
QueueHandle_t audio_tx_queue = NULL;
TaskHandle_t camera_task_handle = NULL;
TaskHandle_t audio_tx_task_handle = NULL;
TaskHandle_t ws_send_task_handle = NULL;
TaskHandle_t udp_send_task_handle = NULL;

void setup() {
    Serial.begin(115200);
    delay(2000);

    // create the global queues
    video_queue = xQueueCreate(2, sizeof(video_frame_t));
    audio_tx_queue = xQueueCreate(16, sizeof(audio_buffer_t));

    // harware stuff
    setupLED();
    setupI2S();
    setupCamera();

    // connect to network
    setupNetwork(ssid, password, udp_server_ip, port);

    // start the tasks; pinned to cores
    xTaskCreatePinnedToCore(camera_task, "camera", 4096, NULL, 4, &camera_task_handle, 1);
    xTaskCreatePinnedToCore(audio_tx_task, "audio_tx", 8196, NULL, 5, &audio_tx_task_handle, 1);
    // xTaskCreatePinnedToCore(websocket_send_task, "ws_send", 4096, NULL, 6, &ws_send_task_handle, 1);
    xTaskCreatePinnedToCore(udp_send_task, "ws_send", 4096, NULL, 6, &udp_send_task_handle, 1);
}

void loop() {
    delay(1000);
}