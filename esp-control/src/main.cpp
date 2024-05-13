#include <Arduino.h>
#include "motor/motor.h"
#include "pin_definitions.h"
#include <HardwareSerial.h>


Motor right_motor(RIGHT_PWM_PIN, RIGHT_PWM_CHANNEL, RIGHT_DIR_PIN, RIGHT_ENCODER_PIN_A, RIGHT_ENCODER_PIN_B, 0);
Motor left_motor(LEFT_PWM_PIN, LEFT_PWM_CHANNEL, LEFT_DIR_PIN, LEFT_ENCODER_PIN_A, LEFT_ENCODER_PIN_B, 1);
HardwareSerial UART1(2);


void setup() {
  right_motor.setup();
  right_motor.set_speed(5);

  left_motor.setup();
  left_motor.set_speed(5);

  // Serial.begin(115200);  // Initialize debugging serial port
  // Initialize UART1 on pins 16 (RX) and 17 (TX)
  UART1.begin(115200, SERIAL_8N1, 16, 17); 
}

// UART managment variables
std::string message_c_str;
size_t split_pos;
int left_speed = 0;
int right_speed = 0;
String encoders_speed_str;

void loop() {
  // Check if data is available
  if (UART1.available()) { 
    String speeds_str = UART1.readString();
    Serial.println("Speeds received: " + speeds_str);
    // transform string to c++ string
    message_c_str = speeds_str.c_str();
    // Find the space separating the two numbers
    split_pos = message_c_str.find(" ");
    // Gather speeds from str message to int values
    left_speed = static_cast<int>(std::stof(message_c_str.substr(0, split_pos)) * 100);
    right_speed = static_cast<int>(std::stof(message_c_str.substr(split_pos + 1)) * 100);

    left_motor.set_speed(left_speed);
    right_motor.set_speed(right_speed);
    encoders_speed_str =  String(left_motor.get_speed()) + " " +  String(right_motor.get_speed());
    UART1.println(encoders_speed_str);
  }

  delay(100);  // Delay for visibility
}