#pragma once

void setupNetwork(const char* ssid, const char* password, const char* server_uri);
void websocket_send_task(void *pvParameters);