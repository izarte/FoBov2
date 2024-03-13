#ifndef MOTOR_H
#define MOTOR_H

#include "motor_definitions.h"

#include <ESP32Encoder.h>
#include <Arduino.h>

class Motor
{
    public:
        Motor(int pwm, int pwm_channel, int dir, int encoderA, int encoderB, bool inverse);
        void setup();
        void set_speed(int speed);

        float get_speed();

    private:
        int pwm_pin;
        int pwm_chan;
        int dir_pin;
        int encoderA;
        int encoderB;
        bool inverse;
        ESP32Encoder encoder;

        unsigned long last_check;
};




#endif