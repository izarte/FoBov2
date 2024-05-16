#include "motor/motor.h"


Motor::Motor(int pwm, int pwm_channel, int dir, int encoderA, int encoderB, bool inverse)
{
    pwm_pin = pwm;
    pwm_chan = pwm_channel;
    dir_pin = dir;
    this->encoderA = encoderA;
    this->encoderB = encoderB;
    this-> inverse = inverse;
}


void Motor::setup()
{
	// Set up pwm signals
	ledcSetup(pwm_chan, FREQ, PWM_RESOLUTION);
	ledcAttachPin(pwm_pin, pwm_chan);
	pinMode(dir_pin, OUTPUT);
	digitalWrite(dir_pin, LOW);

	digitalWrite(dir_pin, LOW);
  	ledcWrite(pwm_chan, 255 * 10 / 100);

    // Attach encoder pins
    encoder.attachHalfQuad(encoderA, encoderB);
    encoder.setCount(0);
    last_check = millis();
}


void Motor::set_speed(int speed)
{
    if (speed > 100)
        speed = 100;
    if (speed < -100)
        speed = -100;

    digitalWrite(dir_pin, speed > 0 ^ inverse);
  	ledcWrite(pwm_chan, 255 * speed / 100);
}


float Motor::get_speed()
{
    int64_t count = encoder.getCount();

    unsigned long elapsed_time = millis() - last_check;

    float elasped_time_s = elapsed_time / 1000.0; 

    float velocity = 0.065625 * count / elasped_time_s;

    last_check = millis();
    encoder.clearCount();

    return velocity;
}