#include <Arduino.h>
#include "motor/motor.h"
#include "pin_definitions.h"
#include <HardwareSerial.h>


Motor right_motor(RIGHT_PWM_PIN, RIGHT_PWM_CHANNEL, RIGHT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
// Motor right_motor(LEFT_PWM_PIN, RIGHT_PWM_CHANNEL, LEFT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
Motor left_motor(LEFT_PWM_PIN, LEFT_PWM_CHANNEL, LEFT_DIR_PIN, LEFT_ENCODER_PIN_A, LEFT_ENCODER_PIN_B, 1);

// Set up UART2
HardwareSerial UART2(2);

void setup()
{
  Serial.begin(115200);
  UART2.begin(115200, SERIAL_8N1, 16, 17);
  right_motor.setup();
  right_motor.set_speed(5);

  left_motor.setup();
  left_motor.set_speed(5);
}

int m = 0;
int i = 1;

// UART managment variables
std::string message_c_str;
size_t split_pos
int left_speed = 0;
int right_speed = 0;
String encoders_speed_str;

void loop() {
    Serial.println("right_motor speed = " + String(right_motor.get_speed()) + " ");
    if (UART2.available()) {
        // Read Rasperry indications
        String speeds_str = UART2.readString();
        Serial.println("Speeds received: " + message);

        // transform string to c++ string
        message_c_str = message.c_str();

        // Find the space separating the two numbers
        split_pos = message_c_str.find(" ");

        // Gather speeds from str message to int valyes
        left_speed = static_cast<int>(std::stof(message_c_str.substr(0, spacePos)) * 100);
        right_speed = static_cast<int>(std::stof(message_c_str.substr(spacePos + 1)) * 100);

        Serial.println("Left speed: " + String(left_speed));
        Serial.println("Right speed: " + String(right_speed));

        left_motor.set_speed(left_speed)
        right_motor.set_speed(right_speed)

        encoders_speed_str =  String(left_motor.get_speed()) + " " +  String(right_motor.get_speed());
        UART2.println(encoders_speed_str);
    }
    delay(10);
  // m += i;
  // if (m > 10)
  //   i = -1;
  // if (m < 0)
  //   i = 1;
  // right_motor.set_speed(m);
  // left_motor.set_speed(m);
}
