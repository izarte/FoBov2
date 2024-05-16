#include <Arduino.h>
#include "motor/motor.h"
#include "pin_definitions.h"
#include <HardwareSerial.h>

Motor right_motor(RIGHT_PWM_PIN, RIGHT_PWM_CHANNEL, RIGHT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
Motor left_motor(LEFT_PWM_PIN, LEFT_PWM_CHANNEL, LEFT_DIR_PIN, LEFT_ENCODER_PIN_A, LEFT_ENCODER_PIN_B, 1);
HardwareSerial UART1(2);


void setup() {
  Serial.begin(115200);
  right_motor.setup();
  right_motor.set_speed(0);

  left_motor.setup();
  left_motor.set_speed(0);

  UART1.begin(1500000, SERIAL_8N1, 16, 17);
}

String inputBuffer = "";
char readenChar;
int separation_idx, left_speed, right_speed;

void loop() {
  while (UART1.available()) {
    char readenChar = (char)UART1.read();
    Serial.print(readenChar);
    inputBuffer += readenChar;
    if (readenChar == '\n') {
      Serial.println("");
      separation_idx = inputBuffer.indexOf(' ');
      left_speed = inputBuffer.substring(0, separation_idx).toInt();
      right_speed = inputBuffer.substring(separation_idx + 1).toInt();

      left_motor.set_speed(left_speed);
      right_motor.set_speed(right_speed);

      String encoders_speed_str = String(left_motor.get_speed()) + " " + String(right_motor.get_speed());
      UART1.println(encoders_speed_str);
      inputBuffer = "";
    }
  }
}
