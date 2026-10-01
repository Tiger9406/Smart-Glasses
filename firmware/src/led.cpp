#include "driver/ledc.h"

#if defined(LED_GPIO_NUM)
#define LED_LEDC_CHANNEL 2
#define LED_LEDC_FREQUENCY 5000
#define LED_LEDC_RESOLUTION 8
#define LED_BRIGHTNESS 128 
#endif

void setupLED() {
#if defined(LED_GPIO_NUM)
    ledcSetup(LED_LEDC_CHANNEL, LED_LEDC_FREQUENCY, LED_LEDC_RESOLUTION);
    ledcAttachPin(LED_GPIO_NUM, LED_LEDC_CHANNEL);
    ledcWrite(LED_LEDC_CHANNEL, 0); 
#endif
}

void setLED(bool on) {
#if defined(LED_GPIO_NUM)
    ledcWrite(LED_LEDC_CHANNEL, on ? LED_BRIGHTNESS : 0);
#endif
}

