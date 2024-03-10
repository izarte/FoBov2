#include <Arduino.h>


#define DIR_PIN 19
#define PWM_PIN 18

#define PWM_CHANNEL 0
#define FREQ 490
#define PWM_RESOLUTION 8


#define EXAMPLE_PCNT_HIGH_LIMIT 100
#define EXAMPLE_PCNT_LOW_LIMIT  -100

#define EXAMPLE_EC11_GPIO_A 0
#define EXAMPLE_EC11_GPIO_B 2








void setup()
{
  ledcSetup(PWM_CHANNEL, FREQ, PWM_RESOLUTION);
  ledcAttachPin(PWM_PIN, PWM_CHANNEL);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(DIR_PIN, LOW);


  pcnt_unit_config_t unit_config = {
    .high_limit = EXAMPLE_PCNT_HIGH_LIMIT,
    .low_limit = EXAMPLE_PCNT_LOW_LIMIT,
  };
  pcnt_unit_handle_t pcnt_unit = NULL;
  ESP_ERROR_CHECK(pcnt_new_unit(&unit_config, &pcnt_unit));
  pcnt_glitch_filter_config_t filter_config = {
      .max_glitch_ns = 1000,
  };
  ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(pcnt_unit, &filter_config));





}

void loop() {
  
  digitalWrite(DIR_PIN, LOW);
  ledcWrite(PWM_CHANNEL, 255 * 10 / 100);
  delay(3000);
  ledcWrite(PWM_CHANNEL, 255 * 30 / 100);
  delay(1000);

  digitalWrite(DIR_PIN, HIGH);
  ledcWrite(PWM_CHANNEL, 255 * 10 / 100);
  delay(3000);
  ledcWrite(PWM_CHANNEL, 255 * 30 / 100);
  delay(1000);
}