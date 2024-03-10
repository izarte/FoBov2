#include "Arduino.h"
#include <ESP32Encoder.h>


ESP32Encoder encoder;

void setup()
{
	Serial.begin(115200);
	// Enable the weak pull down resistors

	//ESP32Encoder::useInternalWeakPullResistors=DOWN;
	// Enable the weak pull up resistors
	ESP32Encoder::useInternalWeakPullResistors = puType::up;

	// use pin 19 and 18 for the first encoder
	encoder.attachHalfQuad(17, 16);
		
	// set starting count value after attaching
	encoder.setCount(0);

	Serial.println("Encoder Start = " + String((int32_t)encoder.getCount()));
}

void loop()
{
	Serial.println("Encoder count = " + String((int32_t)encoder.getCount()) + " ");
	delay(100);
}