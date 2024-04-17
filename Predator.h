#pragma once

#include "CTRNN.h"
#include "Prey.h"

// The Predator class declaration

class Predator {
	public:
		// The constructor
		Predator(double gain, double s_width, double frate,  double handling_time)
		{
			Set(gain, s_width, frate, handling_time);
		};
		Predator() = default;
		// The destructor
		~Predator() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetSensorState(double state) {sensor = state;};

		// Control
        void Set(double gain, double s_width, double frate, double handling_time);
		void Reset(double initpos);
		void Sense(TVector<double> &prey_loc);
		void Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist);

		double pos, gain, sensor, s_width, pastpos,frate, handling_time, 
		handling_counter, munchrate, birthrate, snackflag, condition;
		bool handling;
		Prey prey;
};