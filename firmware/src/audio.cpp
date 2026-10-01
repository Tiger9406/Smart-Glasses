#include "audio.h"
#include "config.h"
#include "driver/i2s.h"

static const int SAMPLE_RATE = 16000;
static const int DMA_BUF_LEN = 512;
static const int DMA_BUF_CNT = 4;
#define I2S_BCLK   19
#define I2S_LRCLK  21
#define I2S_DOUT   20

void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = DMA_BUF_CNT,
        .dma_buf_len = DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK,
        .ws_io_num = I2S_LRCLK,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DOUT
    };
    
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_zero_dma_buffer(I2S_NUM_0);
}

//send audio stuff to global queue for network to worry about
void audio_tx_task(void *pvParameters) {
    size_t bytes_read;
    int32_t i2s_in[512];
    audio_buffer_t audio_buf;
    static int32_t dc_sum = 0;
    // static uint32_t audio_drops = 0;
    
    while (1) {
        i2s_read(I2S_NUM_0, (void*)i2s_in, sizeof(i2s_in), &bytes_read, portMAX_DELAY);
        int samples = bytes_read / sizeof(int32_t);
        audio_buf.count = samples;
        
        for (int i = 0; i < samples; i++) {
            dc_sum = (dc_sum * 15 + i2s_in[i]) >> 4;
            int32_t filtered = i2s_in[i] - dc_sum;
            audio_buf.samples[i] = (int16_t)(filtered >> 14);
        }
        if(xQueueSend(audio_tx_queue, &audio_buf, 0) != pdTRUE){
            // audio_drops++;
            // Serial.printf("[AUDIO] Queue full! Dropped audio chunk. Total drops: %u\n", audio_drops);
            continue;
        }
    }
}