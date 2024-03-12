#include "Arduino.h"
#include <ESP32Encoder.h>


#define PIN_A 17
#define PIN_B 16


#define DIR_PIN 19
#define PWM_PIN 18

#define PWM_CHANNEL 0
#define FREQ 490
#define PWM_RESOLUTION 8


#define EXAMPLE_PCNT_HIGH_LIMIT 100
#define EXAMPLE_PCNT_LOW_LIMIT  -100

#define EXAMPLE_EC11_GPIO_A 0
#define EXAMPLE_EC11_GPIO_B 2



ESP32Encoder encoder;

void setup()
{
	Serial.begin(115200);
	// Enable the weak pull down resistors

	//ESP32Encoder::useInternalWeakPullResistors=DOWN;
	// Enable the weak pull up resistors
	ESP32Encoder::useInternalWeakPullResistors = puType::up;

	// use pin 19 and 18 for the first encoder
	pinMode(PIN_A, INPUT);
	pinMode(PIN_B, INPUT);
	encoder.attachHalfQuad(PIN_A, PIN_B);
		
	// set starting count value after attaching
	encoder.setCount(0);

	Serial.println("Encoder Start = " + String((int32_t)encoder.getCount()));

	// Set up pwm signals
	ledcSetup(PWM_CHANNEL, FREQ, PWM_RESOLUTION);
	ledcAttachPin(PWM_PIN, PWM_CHANNEL);
	pinMode(DIR_PIN, OUTPUT);
	digitalWrite(DIR_PIN, LOW);

	digitalWrite(DIR_PIN, LOW);
  	ledcWrite(PWM_CHANNEL, 255 * 10 / 100);
}

void loop()
{
	// Serial.println(String(digitalRead(PIN_A)) + " " + String(digitalRead(PIN_B)));
	Serial.println("Encoder count = " + String((int32_t)encoder.getCount()) + " ");
	delay(100);
}