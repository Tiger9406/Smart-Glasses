#pragma once

void setupNetwork(const char* ssid, const char* password, const char* server_ip, const int port);
void udp_send_task(void *pvParameters);