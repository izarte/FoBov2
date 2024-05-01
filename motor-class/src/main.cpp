#include <Arduino.h>
#include "motor/motor.h"
#include "pin_definitions.h"


Motor right_motor(RIGHT_PWM_PIN, RIGHT_PWM_CHANNEL, RIGHT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
// Motor right_motor(LEFT_PWM_PIN, RIGHT_PWM_CHANNEL, LEFT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
Motor left_motor(LEFT_PWM_PIN, LEFT_PWM_CHANNEL, LEFT_DIR_PIN, LEFT_ENCODER_PIN_A, LEFT_ENCODER_PIN_B, 1);


void setup()
{
  Serial.begin(115200);
  right_motor.setup();
  right_motor.set_speed(5);

  left_motor.setup();
  left_motor.set_speed(5);
}

int m = 0;
int i = 1;

void loop() {
	Serial.println("right_motor speed = " + String(right_motor.get_speed()) + " ");
	delay(1000);
  // m += i;
  // if (m > 10)
  //   i = -1;
  // if (m < 0)
  //   i = 1;
  // right_motor.set_speed(m);
  // left_motor.set_speed(m);
}
