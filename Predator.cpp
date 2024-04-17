// Eden Forbes
// MinCogEco Predator

#include "Predator.h"
#include "Prey.h"
#include "random.h"
#include "CTRNN.h"

// Constants
const double SpaceSize = 5000;
const double HalfSpace = SpaceSize/2;

// *******
// Control
// *******

// Init the agent
void Predator::Set(double pred_gain, double pred_s_width, double pred_frate, double pred_handling_time)
{
	gain = pred_gain; 
	pos = 0.0;
	pastpos = 0.0;
	sensor = 0.0;
    s_width = pred_s_width;
    frate = pred_frate;
    handling_time = pred_handling_time;
    handling_counter = 0.0;
    handling = false;
    condition = 0.0;
    // interaction rates
    munchrate = 0.0;
    birthrate = 0.0;
    snackflag = 0.0;
}

// Reset the state of the agent
void Predator::Reset(double initpos)
{
	pos = initpos;
	pastpos = initpos;
	sensor = 0.0;
    handling = false;
    handling_counter = 0.0;
}

// Sense 
void Predator::Sense(TVector<double> &prey_loc)
{
    // Sense
	double mindistL = 99999;
    double mindistR = 99999;
    for (int i = 0; i < prey_loc.Size(); i++){
        double d = prey_loc[i] - pos;
        // printf("d = %f\n", d);
        if (d < 0 && d >= -HalfSpace){
            // Closest to the left side, distance is as calculated
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d < 0 && d < -HalfSpace){
            // Closest to the right side, distance is total area + - left side distance 
            d = SpaceSize + d;
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d > 0 && d > HalfSpace){ 
            // Closest to the left side, distance is -total area + right side distance
            d = -SpaceSize + d;
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d > 0 && d <= HalfSpace){
            // Closest to the right side, distance is as calculated
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d == 0){
            d = 0;
        }
        else{
            printf("Prey pos size = %d\n", prey_loc.Size());
            printf("Error in predator sensing\n");
            printf("d = %f\n", d);
        }
    }
    // Cumulate, distance fits in the Gaussian as the difference of the mean (position) and the state (food position)
    // Negate left so negative sensor reading says food left, positive says food right, zero says no food or food on both sides.
	sensor = -2*exp(-(mindistL) * (mindistL) / (2 * s_width * s_width)) + 2*exp(-(mindistR) * (mindistR) / (2 * s_width * s_width));
    // printf("Pred sensor = %f\n", sensor);
}

// Step
void Predator::Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist)
{
    // Remember past position
    pastpos = pos;
	// Update the body position based on the other 2 neurons
    // If still handling previous catch, don't move
    if (handling == true){
        handling_counter += 1;
        if (handling_counter >= handling_time){
            handling = false;
            handling_counter = 0;
        }
    }
    else{
        // Mobile or Sessile?
        if(fabs(sensor) <= 0.01){
            if (condition == 1.0){
                pos += StepSize * gain/2;
            }
            else if (condition == 2.0){
                pos -= StepSize * gain/2;
            }
            else if (condition == 3.0){
                pos = pastpos;
            }
        }
        else {
            pos += StepSize * gain * sensor;
        }
        // Update State if the agent passed food
        if (pastpos < pos){
            if (pos > WorldFood.Size()){
                pos = pos - WorldFood.Size();
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos > pastpos && preylist[i].pos <= WorldFood.Size()){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                    else if (preylist[i].pos >= 0 && preylist[i].pos < pos){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
            else {
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos > pastpos && preylist[i].pos <= pos){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
        }
        if (pastpos > pos){
            if (pos < 0){
                pos = pos + WorldFood.Size();
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos < pastpos && preylist[i].pos >= 0){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                    else if (preylist[i].pos <= WorldFood.Size() && preylist[i].pos > pos){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
            if (pos >= 0) {
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos < pastpos && preylist[i].pos >= pos){
                        preylist[i].state -= preylist[i].state*frate;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
        }
    }
}

