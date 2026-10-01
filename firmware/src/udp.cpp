#include "udp.h"
#include "config.h"
#include <WiFi.h>
#include "lwip/sockets.h"

static int sock = -1;
static struct sockaddr_in dest_addr;

void setupNetwork(const char* ssid, const char* password, const char* server_ip, const int port) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");

    dest_addr.sin_addr.s_addr = inet_addr(server_ip);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);

    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        Serial.println("Unable to create UDP socket");
    }
}

void udp_send_task(void *pvParameters) {
    video_frame_t video_frame;
    audio_buffer_t audio_buf;
    uint8_t *send_buf = NULL;

    while(1) {
        if (sock < 0) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // udp: fire and forget
        while (xQueueReceive(audio_tx_queue, &audio_buf, 0) == pdTRUE) {
            size_t audio_size = audio_buf.count * sizeof(int16_t);
            send_buf = (uint8_t*)malloc(1 + audio_size);
            if (send_buf) {
                send_buf[0] = WS_FRAME_AUDIO_TX;
                memcpy(&send_buf[1], audio_buf.samples, audio_size);
                sendto(sock, send_buf, 1 + audio_size, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
                free(send_buf);
            }
        }

        while (xQueueReceive(video_queue, &video_frame, 0) == pdTRUE) {
            send_buf = (uint8_t*)malloc(1 + video_frame.len);
            if (send_buf) {
                send_buf[0] = WS_FRAME_VIDEO;
                memcpy(&send_buf[1], video_frame.data, video_frame.len);
                sendto(sock, send_buf, 1 + video_frame.len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
                free(send_buf);
            }
            free(video_frame.data); 
        }

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}