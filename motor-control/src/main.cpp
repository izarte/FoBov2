#include <Arduino.h>


#define DIR_PIN 19
#define PWM_PIN 18

#define PWM_CHANNEL 0
#define FREQ 490
#define PWM_RESOLUTION 8

void setup()
{

  ledcSetup(PWM_CHANNEL, FREQ, PWM_RESOLUTION);
  ledcAttachPin(PWM_PIN, PWM_CHANNEL);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(DIR_PIN, LOW);
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