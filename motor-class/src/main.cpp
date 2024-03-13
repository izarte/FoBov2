#include <Arduino.h>
#include "motor/motor.h"
#include "pin_definitions.h"


Motor motor(PWM_PIN, PWM_CHANNEL, DIR_PIN, ENCODER_PIN_A, ENCODER_PIN_B, 0);


void setup()
{
  Serial.begin(115200);
  motor.setup();
  motor.set_speed(20);
}

void loop() {
	Serial.println("Motor speed = " + String(motor.get_speed()) + " ");
	delay(100);
}
