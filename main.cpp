// Eden Forbes
// MinCogEco Script

// ***************************************
// INCLUDES
// ***************************************

#include "Prey.h"
#include "Predator.h"
#include "random.h"
#include "TSearch.h"
#include <iostream>
#include <iomanip> 
#include <vector>
#include <string>
#include <list>

// ================================================
// A. PARAMETERS & GEN-PHEN MAPPING
// ================================================

// Run constants
// Make sure SpaceSize is also specified in the Prey.cpp and Predator.cpp files
const int SpaceSize = 5000;
const int finalCC = 2;
const int CC = 2;
const int minCC = 0;
const int maxCC = 9;
// 0-Indexed (0 = 1)
const int start_prey = 0;
const int start_pred = 0;

// Time Constants
// Evolution Step Size:
const double StepSize = 0.1;
// Analysis Step Size:
const double BTStepSize = 0.1;
// Evolution Run Time:
const double RunDuration = 10000;
// Behavioral Trace Run Time:
const double PlotDuration = 3000;
// EcoRate Collection Run Time:
const double RateDuration = 50000;
// Sensory Sample Run Time:
const double SenseDuration = 2000; // 2000

// EA params
const int POPSIZE = 200;
const int GENS = 500;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;
// Number of trials per trial type (there are maxCC+1 * 2 trial types)
const double n_trials = 2.0;

// Nervous system params
const int prey_netsize = 3; 
// weight range 
const double WR = 16.0;
// sensor range
const double SR = 20.0;
// bias range
const double BR = 16.0;
// time constant min max
const double TMIN = 0.5;
const double TMAX = 20.0;
// Weights + TCs & Biases + SensorWeights
const int VectSize = (prey_netsize*prey_netsize + 2*prey_netsize + 3*prey_netsize);

// Producer Parameters
const double G_Rate = 0.001*StepSize;
const double BT_G_Rate = 0.001*BTStepSize;
// Prey Sensory Parameters
const double prey_gain = 3.5;
const double prey_s_width = 100.0;

// Prey Metabolic Parameters
// prey_loss_scalar MUST be greater than 1 to prevent strafing behaviors
const double prey_loss_scalar = 3.0;
const double prey_frate = 0.15;
const double prey_feff = 0.1;
const double prey_repo = 1.5;
const double prey_movecost = 0.0;
const double prey_b_thresh = 3.0;
const double prey_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*StepSize))) * prey_loss_scalar;
const double prey_BT_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*BTStepSize))) * prey_loss_scalar;

// Predator Sensory Parameters 
const double pred_gain = 3.0;
const double pred_s_width = 110.0;

// Predator Metabolic Parameters
const double pred_frate = 1.0;
const double pred_handling_time = 10.0/StepSize;
const double pred_BT_handling_time = 10.0/BTStepSize;
// For Predator Condition:
// 1.0 drift left
// 2.0 drift right
// 3.0 stay still
const double pred_condition = 2.0;

// File Names
const string bestgen_string = "bestgenind.dat";
const string fitness_string = "fitnessind.dat";
const string seed_string = "seedind.dat";

// ------------------------------------
// Genotype-Phenotype Mapping Function
// ------------------------------------

void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Prey Time-constants
	for (int i = 1; i <= prey_netsize; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Prey Bias
	for (int i = 1; i <= prey_netsize; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Prey Weights
	for (int i = 1; i <= prey_netsize; i++) {
		for (int j = 1; j <= prey_netsize; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
	// Prey Sensor Weights
	for (int i = 1; i <= 3*prey_netsize; i++) {
        phen(k) = MapSearchParameter(gen(k), -SR, SR);
		k++;
	}
}

// ================================================
// B. TASK ENVIRONMENT & FITNESS FUNCTION
// ================================================
double Coexist(TVector<double> &genotype, RandomState &rs) 
{
    // Set running outcome variable
    double outcome = 99999999999.0;
    // Translate genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Initialize Prey & Predator agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
    Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_handling_time);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    int j = k;
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    // Set Trial Structure
    TVector<TVector<double> > trials(0,-1);
    for (int i = minCC; i <= maxCC; i++){
        for (int j = 0; j < n_trials; j++){
            TVector<double> trialP1(0,1);
            trialP1[0] = i;
            trialP1[1] = 1.0;
            TVector<double> trialP2(0,1);
            trialP2[0] = i;
            trialP2[1] = 2.0;
            trials.SetBounds(0, trials.Size());
            trials[trials.Size()-1] = trialP1;
            trials.SetBounds(0, trials.Size());
            trials[trials.Size()-1] = trialP2;
        }
    }
    // Run Simulation
    for (int trial = 0; trial < trials.Size(); trial++){
        // Set Predator Condition for Trial
        // Start prey 0-indexed
        double CC = trials[trial][0] * (start_prey+1);
        Agent2.condition = trials[trial][1];;
        // Reset Prey agent, randomize its location
        Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        // Seed preylist with starting population
        TVector<Prey> preylist(0,0);
        preylist[0] = Agent1;
        for (int i = 0; i < start_prey; i++){
            Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
            // Reset Prey agent, randomize its location
            newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
            // Copy over nervous system
            newprey.NervousSystem = Agent1.NervousSystem;
            newprey.sensorweights = Agent1.sensorweights;
            // Add to preylist
            preylist.SetBounds(0, preylist.Size());
            preylist[preylist.Size()-1] = newprey;
        }
        // Seed predlist with starting population
        TVector<Predator> predlist(0,0);
        predlist[0] = Agent2;
        for (int i = 0; i < start_pred; i++){
            Predator newpred(pred_gain, pred_s_width, pred_frate, pred_handling_time);
            // Reset Predator agent, randomize its location
            newpred.Reset(rs.UniformRandomInteger(0,SpaceSize));
            newpred.condition = pred_condition;
            // Add to predlist
            predlist.SetBounds(0, predlist.Size());
            predlist[predlist.Size()-1] = newpred;
        }
        // Initialize Producers, fill world to carrying capacity
        TVector<double> food_pos;
        TVector<double> WorldFood(1, SpaceSize);
        WorldFood.FillContents(0.0);
        for (int i = 0; i <= CC; i++){
            int f = rs.UniformRandomInteger(1,SpaceSize);
            WorldFood[f] = 1.0;
            food_pos.SetBounds(0, food_pos.Size());
            food_pos[food_pos.Size()-1] = f;
        }
        // Set Clocks & trial outcome variables
        double clock = 0.0;
        double prey_snacks = 0.0;
        double pred_snacks = 0.0;
        double prey_outcome = RunDuration;
        double pred_outcome = RunDuration;
        double running_pop = 0.0;
        double last_state = 0.0;

        // Run a Trial
        for (double time = 0; time < RunDuration; time += StepSize){
            // Remove any consumed food from food list
            TVector<double> dead_food(0,-1);
            for (int i = 0; i < food_pos.Size(); i++){
                if (WorldFood[food_pos[i]] <= 0){
                    dead_food.SetBounds(0, dead_food.Size());
                    dead_food[dead_food.Size()-1] = food_pos[i];
                }
            }
            if (dead_food.Size() > 0){
                for (int i = 0; i < dead_food.Size(); i++){
                    food_pos.RemoveFood(dead_food[i]);
                    food_pos.SetBounds(0, food_pos.Size()-2);
                }
            }
            // Chance for new food to grow
            // Carrying capacity is 0 indexed, add 1 for true amount
            for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            // Update Prey Positions
            TVector<double> prey_pos;
            for (int i = 0; i < preylist.Size(); i++){
                prey_pos.SetBounds(0, prey_pos.Size());
                prey_pos[prey_pos.Size()-1] = preylist[i].pos;
            }
            // Predator Sense & Step
            TVector<Predator> newpredlist;
            TVector<int> preddeaths;
            for (int i = 0; i < predlist.Size(); i++){
                predlist[i].Sense(prey_pos);
                predlist[i].Step(StepSize, WorldFood, preylist);
                if (predlist[i].snackflag > 0){
                    pred_snacks += predlist[i].snackflag;
                    predlist[i].snackflag = 0;
                }
            }
            // Update Predator Positions
            TVector<double> pred_pos;
            for (int i = 0; i < predlist.Size(); i++){
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predlist[i].pos;
            }
            // Prey Sense & Step
            TVector<Prey> newpreylist;
            TVector<int> preydeaths;
            for (int i = 0; i < preylist.Size(); i++){
                preylist[i].Sense(food_pos, pred_pos);
                preylist[i].Step(StepSize, WorldFood);
                if (preylist[i].snackflag > 0){
                    prey_snacks += preylist[i].snackflag;
                    preylist[i].snackflag = 0;
                }
                // ONLY WITH IND
                // if (preylist[i].state > 3.0){
                //     preylist[i].state = 3.0;
                // }
                // // ONLY WITH POP
                // if (preylist[i].birth == true){
                //     preylist[i].state = preylist[i].state - prey_repo;
                //     preylist[i].birth = false;
                //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
                //     newprey.NervousSystem = preylist[i].NervousSystem;
                //     newprey.sensorweights = preylist[i].sensorweights;
                //     newprey.Reset(preylist[i].pos+2, prey_repo);
                //     newpreylist.SetBounds(0, newpreylist.Size());
                //     newpreylist[newpreylist.Size()-1] = newprey;
                // }
                if (preylist[i].death == true){
                    preydeaths.SetBounds(0, preydeaths.Size());
                    preydeaths[preydeaths.Size()-1] = i;
                }
            }
            // Update clocks
            clock += StepSize;
            // Update prey list with new prey list and deaths
            if (preydeaths.Size() > 0){
                for (int i = 0; i <= preydeaths.Size()-1; i++){
                    preylist.RemoveItem(preydeaths[i]);
                    preylist.SetBounds(0, preylist.Size()-2);
                }
            }
            if (newpreylist.Size() > 0){
                for (int i = 0; i <= newpreylist.Size()-1; i++){
                    preylist.SetBounds(0, preylist.Size());
                    preylist[preylist.Size()-1] = newpreylist[i];
                }
            }
            // Check for prey population collapse
            if (preylist.Size() <= 0){
                break;
            }
            // Reset lists for next step
            else{
                last_state = preylist[0].state;
                running_pop += preylist.Size();
                newpreylist.~TVector();
                preydeaths.~TVector();
                newpredlist.~TVector();
                preddeaths.~TVector();
                prey_pos.~TVector();
                pred_pos.~TVector();
                dead_food.~TVector();
            }
        }
        // Fitness part 1 is proportion of RunTime survived
        double runmeasure = (clock/RunDuration);
        // FOR POPULATIONS
        // Fitness part 2 is average population across the run
        // double popmeasure = (running_pop/(RunDuration/StepSize));
        // FOR INDIVIDUALS
        // Fitness part 2 is end state
        double popmeasure = last_state;
        // Generate fitness
        double fitmeasure = runmeasure + popmeasure/(10*CC);
        // Keep minimum fitness value across trials
        if (fitmeasure < outcome){
            outcome = fitmeasure;
        }
    }
    return outcome;
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int CCTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= 1.0) return 1;
	else return 0;
}

int EndTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= 100.0) return 1;
	else return 0;
}

void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
    BestIndividualFile << setprecision(32);
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ================================================
// D. ANALYSIS FUNCTIONS
// ================================================
// ------------------------------------
// Behavioral Traces
// ------------------------------------

double BehavioralTracesCoexist (TVector<double> &genotype, RandomState &rs, double condition, int agent) 
{
    // Start output files
    int i = 4;
    std::string pyfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pdfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pypfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pdpfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string ffile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string fpfile = "menagerie/IndBatch2/analysis_results/ns_";
    pyfile += std::to_string(agent);
    pdfile += std::to_string(agent);
    pypfile += std::to_string(agent);
    pdpfile += std::to_string(agent);
    ffile += std::to_string(agent);
    fpfile += std::to_string(agent);
    pyfile += "/xx_prey_pos.dat";
    pdfile += "/xx_pred_pos.dat";
    pypfile += "/xx_prey_pop.dat";
    pdpfile += "/xx_pred_pop.dat";
    ffile += "/xx_food_pos.dat";
    fpfile += "/xx_food_pop.dat";
    ofstream preyfile(pyfile);
    ofstream predfile(pdfile);
    ofstream preypopfile(pypfile);
    ofstream predpopfile(pdpfile);
    ofstream foodfile(ffile);
    ofstream foodpopfile(fpfile);

	// ofstream preyfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_prey_pos.dat", agent);
    // ofstream predfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_pred_pos.dat", agent);
    // ofstream preypopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_prey_pop.dat", agent);
    // ofstream predpopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_pred_pop.dat", agent);
	// ofstream foodfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_food_pos.dat", agent);
    // ofstream foodpopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_food_pop.dat", agent);
    // printf("Starting run for agent %d\n", agent);
    // Set running outcome
    double outcome = 99999999999.0;
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
    Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
    // Set nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    // Run Simulation
    double prey_outcome = 99999999999.0;
    double pred_outcome = 99999999999.0;
    Agent2.condition = condition;
    // Reset Agents & Vectors
    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize));

    // Seed preylist with starting population
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    for (int i = 0; i < start_prey; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.0);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
    }
    // Seed predlist with starting population
    TVector<Predator> predlist(0,0);
    predlist[0] = Agent2;
    for (int i = 0; i < start_pred; i++){
        Predator newpred(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
        newpred.Reset(rs.UniformRandomInteger(0,SpaceSize));
        newpred.condition = pred_condition;
        predlist.SetBounds(0, predlist.Size());
        predlist[predlist.Size()-1] = newpred;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    // Run Simulation
    for (double time = 0; time < PlotDuration; time += BTStepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= BT_G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        // Predator Sense & Step
        TVector<Predator> newpredlist;
        TVector<int> preddeaths;
        for (int i = 0; i < predlist.Size(); i++){
            predlist[i].Sense(prey_pos);
            predlist[i].Step(BTStepSize, WorldFood, preylist);
        }
        // Update Predator Positions
        TVector<double> pred_pos;
        for (int i = 0; i < predlist.Size(); i++){
            pred_pos.SetBounds(0, pred_pos.Size());
            pred_pos[pred_pos.Size()-1] = predlist[i].pos;
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos);
            preylist[i].Step(BTStepSize, WorldFood);
            // FOR POPS ONLY
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
                Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
                newprey.NervousSystem = preylist[i].NervousSystem;
                newprey.sensorweights = preylist[i].sensorweights;
                newprey.Reset(preylist[i].pos+2, prey_repo);
                newpreylist.SetBounds(0, newpreylist.Size());
                newpreylist[newpreylist.Size()-1] = newprey;
            }
            // // FOR INDS ONLY
            // if (preylist[i].state > 3.0){
            //     preylist[i].state = 3.0;
            // }
            if (preylist[i].death == true){
                preydeaths.SetBounds(0, preydeaths.Size());
                preydeaths[preydeaths.Size()-1] = i;
            }
        }
        // Update prey list with new prey list and deaths
        if (preydeaths.Size() > 0){
            for (int i = 0; i < preydeaths.Size(); i++){
                preylist.RemoveItem(preydeaths[i]);
                preylist.SetBounds(0, preylist.Size()-2);
            }
        }
        if (newpreylist.Size() > 0){
            for (int i = 0; i < newpreylist.Size(); i++){
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newpreylist[i];
            }
        }
        // Save
        preyfile << prey_pos << endl;
        preypopfile << preylist.Size() << " ";
        predfile << pred_pos << endl;
        predpopfile << predlist.Size() << " ";
        foodfile << food_pos << endl;
        double foodsum = 0.0;
        for (int i = 0; i < food_pos.Size(); i++){
            foodsum += WorldFood[food_pos[i]];
        }
        foodpopfile << foodsum << " ";
        // Check Population Collapse
        if (preylist.Size() <= 0){
            break;
        }
        else{
            newpreylist.~TVector();
            preydeaths.~TVector();
            newpredlist.~TVector();
            preddeaths.~TVector();
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
        }
    }
    preyfile.close();
    preypopfile.close();
    predfile.close();
    predpopfile.close();
	foodfile.close();
    foodpopfile.close();
    // Save best phenotype
    // bestphen << phenotype << endl;
    return 0;
}

// ------------------------------------
// Interaction Rate Data Functions
// ------------------------------------
void DeriveLambdaH(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
{
    ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH.dat");
    // for (int i = 0; i <= 0; i++){
        TVector<TVector<double> > lambH;
        for (int j = 0; j <= maxCC; j++){
            TVector<double> lambHcc;
            for (int k = 0; k <= samplesize; k++){
                int carrycapacity = j;
                // Fill World to Carrying Capacity
                TVector<double> food_pos;
                TVector<double> WorldFood(1, SpaceSize);
                WorldFood.FillContents(0.0);
                for (int i = 0; i <= carrycapacity; i++){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
                // Seed preylist with starting population
                TVector<Prey> preylist(0,0);
                TVector<double> prey_pos;
                preylist[0] = prey;
                // for (int i = 0; i < j; i++){
                //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
                //     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
                //     newprey.NervousSystem = prey.NervousSystem;
                //     newprey.sensorweights = prey.sensorweights;
                //     preylist.SetBounds(0, preylist.Size());
                //     preylist[preylist.Size()-1] = newprey;
                //     }
                // Make dummy predator list
                TVector<double> pred_pos(0,-1);
                double munch_count = 0;
                for (double time = 0; time < RateDuration; time += BTStepSize){
                    // Remove chomped food from food list
                    TVector<double> dead_food(0,-1);
                    for (int i = 0; i < food_pos.Size(); i++){
                        if (WorldFood[food_pos[i]] <= 0){
                            dead_food.SetBounds(0, dead_food.Size());
                            dead_food[dead_food.Size()-1] = food_pos[i];
                        }
                    }
                    if (dead_food.Size() > 0){
                        for (int i = 0; i < dead_food.Size(); i++){
                            food_pos.RemoveFood(dead_food[i]);
                            food_pos.SetBounds(0, food_pos.Size()-2);
                        }
                    }
                    // Carrying capacity is 0 indexed, add 1 for true amount
                    for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
                        double c = rs.UniformRandom(0,1);
                        if (c <= BT_G_Rate){
                            int f = rs.UniformRandomInteger(1,SpaceSize);
                            WorldFood[f] = 1.0;
                            food_pos.SetBounds(0, food_pos.Size());
                            food_pos[food_pos.Size()-1] = f;
                        }
                    }
                    for (int i = 0; i < preylist.Size(); i++){
                        // Prey Sense & Step
                        preylist[i].Sense(food_pos, pred_pos);
                        preylist[i].Step(BTStepSize, WorldFood);
                        // Check Births
                        if (preylist[i].birth == true){
                            preylist[i].state = preylist[i].state - prey_repo;
                            preylist[i].birth = false;
                        }
                        // Check Deaths
                        if (preylist[i].death == true){
                            preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
                            preylist[i].death = false;
                        }
                        // Check # of times food crossed
                        if (time > transient){
                            munch_count += preylist[i].snackflag;
                            preylist[i].snackflag = 0.0;
                        }
                    }
                }
                double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
                lambHcc.SetBounds(0, lambHcc.Size());
                lambHcc[lambHcc.Size()-1] = munchrate;
            }
        //     lambH.SetBounds(0, lambH.Size());
        //     lambH[lambH.Size()-1] = lambHcc;
        // }
        lambHfile << lambHcc << endl;
        lambHcc.~TVector();
    }
    // Save
    lambHfile.close();
}

void DeriveLambdaH2(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
{
    ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH3.dat");
    // for (int i = 0; i <= 0; i++){
        TVector<TVector<double> > lambH;
        for (int j = -1; j <= maxprey; j++){
            TVector<double> lambHcc;
            for (int k = 0; k <= samplesize; k++){
                int carrycapacity = 29;
                // Fill World to Carrying Capacity
                TVector<double> food_pos;
                TVector<double> WorldFood(1, SpaceSize);
                WorldFood.FillContents(0.0);
                for (int i = 0; i <= carrycapacity; i++){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
                // Seed preylist with starting population
                TVector<Prey> preylist(0,0);
                TVector<double> prey_pos;
                preylist[0] = prey;
                for (int i = 0; i < j; i++){
                    Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
                    newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
                    newprey.NervousSystem = prey.NervousSystem;
                    newprey.sensorweights = prey.sensorweights;
                    preylist.SetBounds(0, preylist.Size());
                    preylist[preylist.Size()-1] = newprey;
                    }
                // Make dummy predator list
                TVector<double> pred_pos(0,-1);
                double munch_count = 0;
                for (double time = 0; time < RateDuration; time += BTStepSize){
                    // Remove chomped food from food list
                    TVector<double> dead_food(0,-1);
                    for (int i = 0; i < food_pos.Size(); i++){
                        if (WorldFood[food_pos[i]] <= 0){
                            dead_food.SetBounds(0, dead_food.Size());
                            dead_food[dead_food.Size()-1] = food_pos[i];
                        }
                    }
                    if (dead_food.Size() > 0){
                        for (int i = 0; i < dead_food.Size(); i++){
                            food_pos.RemoveFood(dead_food[i]);
                            food_pos.SetBounds(0, food_pos.Size()-2);
                        }
                    }
                    // Carrying capacity is 0 indexed, add 1 for true amount
                    for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
                        double c = rs.UniformRandom(0,1);
                        if (c <= BT_G_Rate){
                            int f = rs.UniformRandomInteger(1,SpaceSize);
                            WorldFood[f] = 1.0;
                            food_pos.SetBounds(0, food_pos.Size());
                            food_pos[food_pos.Size()-1] = f;
                        }
                    }
                    for (int i = 0; i < preylist.Size(); i++){
                        // Prey Sense & Step
                        preylist[i].Sense(food_pos, pred_pos);
                        preylist[i].Step(BTStepSize, WorldFood);
                        // Check Births
                        if (preylist[i].birth == true){
                            preylist[i].state = preylist[i].state - prey_repo;
                            preylist[i].birth = false;
                        }
                        // Check Deaths
                        if (preylist[i].death == true){
                            preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
                            preylist[i].death = false;
                        }
                        // Check # of times food crossed
                        if (time > transient){
                            munch_count += preylist[i].snackflag;
                            preylist[i].snackflag = 0.0;
                        }
                    }
                }
                double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
                lambHcc.SetBounds(0, lambHcc.Size());
                lambHcc[lambHcc.Size()-1] = munchrate;
            }
        //     lambH.SetBounds(0, lambH.Size());
        //     lambH[lambH.Size()-1] = lambHcc;
        // }
        lambHfile << lambHcc << endl;
        lambHcc.~TVector();
    }
    // Save
    lambHfile.close();
}

void DeriveRR(RandomState &rs, double &testCC, int &samplesize)
{
    ofstream RRfile("menagerie/IndBatch2/analysis_results/ns_15/RR.dat");
    for (int r = -1; r <= testCC; r++){
        TVector<double> RR;
        for (int k = 0; k <= samplesize; k++){
            double counter = 0;
            for (double time = 0; time < RateDuration; time += BTStepSize){
                for (int i = 0; i < ((testCC+1) - (r+1)); i++){
                    double c = rs.UniformRandom(0,1);
                    if (c <= BT_G_Rate){
                        int f = rs.UniformRandomInteger(1,SpaceSize);
                        counter += 1;
                    }
                }
            }
            RR.SetBounds(0, RR.Size());
            RR[RR.Size()-1] = counter/(RateDuration/BTStepSize);
        }
        RRfile << RR << endl;
        RR.~TVector();
    }
    // Save
    RRfile.close();
}

void DeriveLambdaP(Prey &prey, Predator &predator, RandomState &rs, double &maxprey, int &samplesize, double &transient)
{
    ofstream lambCfile("menagerie/IndBatch2/analysis_results/ns_15/lambC.dat");
    for (int j = -1; j<=maxprey; j++)
    {   
        TVector<double> lambC;
        for (int k = 0; k<=samplesize; k++){
            // Fill World to Carrying Capacity
            TVector<double> food_pos;
            TVector<double> WorldFood(1, SpaceSize);
            WorldFood.FillContents(0.0);
            for (int i = 0; i <= CC; i++){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
            // Seed preylist with starting population
            TVector<Prey> preylist(0,0);
            TVector<double> prey_pos;
            preylist[0] = prey;
            for (int i = 0; i < j; i++){
                Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
                newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
                newprey.NervousSystem = prey.NervousSystem;
                newprey.sensorweights = prey.sensorweights;
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newprey;
                }
            double munch_count = 0;
            for (double time = 0; time < RateDuration; time += BTStepSize){
                // Remove chomped food from food list
                TVector<double> dead_food(0,-1);
                for (int i = 0; i < food_pos.Size(); i++){
                    if (WorldFood[food_pos[i]] <= 0){
                        dead_food.SetBounds(0, dead_food.Size());
                        dead_food[dead_food.Size()-1] = food_pos[i];
                    }
                }
                if (dead_food.Size() > 0){
                    for (int i = 0; i < dead_food.Size(); i++){
                        food_pos.RemoveFood(dead_food[i]);
                        food_pos.SetBounds(0, food_pos.Size()-2);
                    }
                }
                // Carrying capacity is 0 indexed, add 1 for true amount
                for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
                    double c = rs.UniformRandom(0,1);
                    if (c <= BT_G_Rate){
                        int f = rs.UniformRandomInteger(1,SpaceSize);
                        WorldFood[f] = 1.0;
                        food_pos.SetBounds(0, food_pos.Size());
                        food_pos[food_pos.Size()-1] = f;
                    }
                }
                // Prey Sense & Step
                TVector<Prey> newpreylist;
                TVector<int> deaths;
                TVector<double> prey_pos;
                TVector<double> pred_pos;
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predator.pos;
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].death == true){
                        preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
                    }
                    else{
                        preylist[i].Sense(food_pos, pred_pos);
                        preylist[i].Step(BTStepSize, WorldFood);
                    }
                }
                for (int i = 0; i <= preylist.Size()-1; i++){
                    prey_pos.SetBounds(0, prey_pos.Size());
                    prey_pos[prey_pos.Size()-1] = preylist[i].pos;
                }

                // Predator Sense & Step
                predator.Sense(prey_pos);
                predator.Step(BTStepSize, WorldFood, preylist);
                // Check # of times food crossed
                if(time > transient){
                    munch_count += predator.snackflag;
                    predator.snackflag = 0.0;
                }
            }

            double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
            lambC.SetBounds(0, lambC.Size());
            lambC[lambC.Size()-1] = munchrate;
        }
        lambCfile << lambC << endl;
        lambC.~TVector();
    }
    // Save
    lambCfile.close();
}

void CollectEcoRates(TVector<double> &genotype, RandomState &rs)
{
    ofstream erates("menagerie/IndBatch2/analysis_results/ns_15/ecosystem_rates.dat");
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
    Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
    Agent2.condition = pred_condition;
    // Set nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    // Save Growth Rates
    // Max growth rate of producers is the chance of a new plant coming in on a given time step
    double systemcc = CC+1; // 0 indexed
    double rr = BT_G_Rate;
    erates << rr << " ";
    erates << systemcc << " ";
    erates << Agent1.frate << " ";
    erates << Agent1.feff << " ";
    erates << Agent1.metaloss << " ";
    erates << Agent2.frate << " ";
    // erates << Agent2.feff << " ";
    // erates << Agent2.metaloss << " ";
    erates.close();

    // Set Sampling Range & Frequency
    double maxCC = 300;
    double maxprey = 60;
    double transient = 100.0;
    int samplesize = 10;
    double testCC = 29;
    // Collect rr at testCC
    // printf("Collecting Growth Rate at Test Carrying Capacity\n");
    DeriveRR(rs, testCC, samplesize);
    // Collect Prey Lambda & r
    // printf("Collecting Prey rates\n");
    // DeriveLambdaH(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
    // DeriveLambdaH2(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
    // Collect Predator Lambda & r
    // printf("Collecting Predator rates\n");
    // DeriveLambdaP(Agent1, Agent2, rs, maxprey, samplesize, transient);
}

// ------------------------------------
// Sensory Sample Functions
// ------------------------------------
double SSCoexist(TVector<double> &genotype, RandomState &rs, double condition, int agent) 
{
    // Start output files
    int i = 4;
    ofstream SSfile("menagerie/IndBatch2/analysis_results/ns_15/SenSxx.dat");
    std::string pyfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pdfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pypfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string pdpfile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string ffile = "menagerie/IndBatch2/analysis_results/ns_";
    std::string fpfile = "menagerie/IndBatch2/analysis_results/ns_";
    pyfile += std::to_string(agent);
    pdfile += std::to_string(agent);
    pypfile += std::to_string(agent);
    pdpfile += std::to_string(agent);
    ffile += std::to_string(agent);
    fpfile += std::to_string(agent);
    pyfile += "/xx_prey_pos.dat";
    pdfile += "/xx_pred_pos.dat";
    pypfile += "/xx_prey_pop.dat";
    pdpfile += "/xx_pred_pop.dat";
    ffile += "/xx_food_pos.dat";
    fpfile += "/xx_food_pop.dat";
    ofstream preyfile(pyfile);
    ofstream predfile(pdfile);
    ofstream preypopfile(pypfile);
    ofstream predpopfile(pdpfile);
    ofstream foodfile(ffile);
    ofstream foodpopfile(fpfile);
    TVector<double> FS;
    TVector<double> PS;
    TVector<double> SS;
    TVector<double> NO1;
    TVector<double> N1FS;
    TVector<double> N1PS;
    TVector<double> N1SS;
    TVector<double> NO2;
    TVector<double> N2FS;
    TVector<double> N2PS;
    TVector<double> N2SS;
    TVector<double> NO3;
    TVector<double> N3FS;
    TVector<double> N3PS;
    TVector<double> N3SS;
    TVector<double> mov;

	// ofstream preyfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_prey_pos.dat", agent);
    // ofstream predfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_pred_pos.dat", agent);
    // ofstream preypopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_prey_pop.dat", agent);
    // ofstream predpopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_pred_pop.dat", agent);
	// ofstream foodfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_food_pos.dat", agent);
    // ofstream foodpopfile("menagerie/PopBatch4/analysis_results/ns_%d/c2_food_pop.dat", agent);
    // printf("Starting run for agent %d\n", agent);
    // Set running outcome
    double outcome = 99999999999.0;
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
    Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
    // Set nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    // Run Simulation
    double prey_outcome = 99999999999.0;
    double pred_outcome = 99999999999.0;
    Agent2.condition = condition;
    // Reset Agents & Vectors
    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize));

    // Seed preylist with starting population
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    for (int i = 0; i < start_prey; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.0);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
    }
    // Seed predlist with starting population
    TVector<Predator> predlist(0,0);
    predlist[0] = Agent2;
    for (int i = 0; i < start_pred; i++){
        Predator newpred(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
        newpred.Reset(rs.UniformRandomInteger(0,SpaceSize));
        newpred.condition = pred_condition;
        predlist.SetBounds(0, predlist.Size());
        predlist[predlist.Size()-1] = newpred;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    // Run Simulation
    for (double time = 0; time < PlotDuration; time += BTStepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= BT_G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        // Predator Sense & Step
        TVector<Predator> newpredlist;
        TVector<int> preddeaths;
        for (int i = 0; i < predlist.Size(); i++){
            predlist[i].Sense(prey_pos);
            predlist[i].Step(BTStepSize, WorldFood, preylist);
        }
        // Update Predator Positions
        TVector<double> pred_pos;
        for (int i = 0; i < predlist.Size(); i++){
            pred_pos.SetBounds(0, pred_pos.Size());
            pred_pos[pred_pos.Size()-1] = predlist[i].pos;
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos);
            FS.SetBounds(0, SS.Size());
            FS[FS.Size()-1] = preylist[i].f_sensor;
            N1FS.SetBounds(0, N1FS.Size());
            N1FS[N1FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[1];
            N2FS.SetBounds(0, N2FS.Size());
            N2FS[N2FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[4];
            N3FS.SetBounds(0, N3FS.Size());
            N3FS[N3FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[7];
            PS.SetBounds(0, PS.Size());
            PS[PS.Size()-1] = preylist[i].p_sensor;
            N1PS.SetBounds(0, N1PS.Size());
            N1PS[N1PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[2];
            N2PS.SetBounds(0, N2PS.Size());
            N2PS[N2PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[5];
            N3PS.SetBounds(0, N3PS.Size());
            N3PS[N3PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[8];
            SS.SetBounds(0, SS.Size());
            SS[SS.Size()-1] = preylist[i].state;
            N1SS.SetBounds(0, N1SS.Size());
            N1SS[N1SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[3];
            N2SS.SetBounds(0, N2SS.Size());
            N2SS[N2SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[6];
            N3SS.SetBounds(0, N3SS.Size());
            N3SS[N3SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[9];

            preylist[i].Step(BTStepSize, WorldFood);
            NO1.SetBounds(0, NO1.Size());
            NO1[NO1.Size()-1] = preylist[i].NervousSystem.NeuronOutput(1);
            NO2.SetBounds(0, NO2.Size());
            NO2[NO2.Size()-1] = preylist[i].NervousSystem.NeuronOutput(2);
            NO3.SetBounds(0, NO3.Size());
            NO3[NO3.Size()-1] = preylist[i].NervousSystem.NeuronOutput(3);
            mov.SetBounds(0, mov.Size());
            mov[mov.Size()-1] = (preylist[i].NervousSystem.NeuronOutput(2) - preylist[i].NervousSystem.NeuronOutput(1));

            // FOR POPS ONLY
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
                Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
                newprey.NervousSystem = preylist[i].NervousSystem;
                newprey.sensorweights = preylist[i].sensorweights;
                newprey.Reset(preylist[i].pos+2, prey_repo);
                newpreylist.SetBounds(0, newpreylist.Size());
                newpreylist[newpreylist.Size()-1] = newprey;
            }
            // // FOR INDS ONLY
            // if (preylist[i].state > 3.0){
            //     preylist[i].state = 3.0;
            // }
            if (preylist[i].death == true){
                preydeaths.SetBounds(0, preydeaths.Size());
                preydeaths[preydeaths.Size()-1] = i;
            }
        }
        // Update prey list with new prey list and deaths
        if (preydeaths.Size() > 0){
            for (int i = 0; i < preydeaths.Size(); i++){
                preylist.RemoveItem(preydeaths[i]);
                preylist.SetBounds(0, preylist.Size()-2);
            }
        }
        if (newpreylist.Size() > 0){
            for (int i = 0; i < newpreylist.Size(); i++){
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newpreylist[i];
            }
        }
        // Save
        preyfile << prey_pos << endl;
        preypopfile << preylist.Size() << " ";
        predfile << pred_pos << endl;
        predpopfile << predlist.Size() << " ";
        foodfile << food_pos << endl;
        double foodsum = 0.0;
        for (int i = 0; i < food_pos.Size(); i++){
            foodsum += WorldFood[food_pos[i]];
        }
        foodpopfile << foodsum << " ";
        // Check Population Collapse
        if (preylist.Size() <= 0){
            break;
        }
        else{
            newpreylist.~TVector();
            preydeaths.~TVector();
            newpredlist.~TVector();
            preddeaths.~TVector();
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
        }
    }
    preyfile.close();
    preypopfile.close();
    predfile.close();
    predpopfile.close();
	foodfile.close();
    foodpopfile.close();

    SSfile << FS << endl << N1FS << endl << N2FS << endl << N3FS << endl;
    SSfile << PS << endl << N1PS << endl << N2PS << endl << N3PS << endl;
    SSfile << SS << endl << N1SS << endl << N2SS << endl << N3SS << endl; 
    SSfile << NO1 << endl << NO2 << endl << NO3 << endl << mov << endl;

    SSfile.close();
    // Save best phenotype
    // bestphen << phenotype << endl;
    return 0;
}

void SensorySample(TVector<double> &genotype, RandomState &rs)
{
    ofstream SSfile("menagerie/IndBatch2/analysis_results/ns_15/SenS2.dat");
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
    Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
    // Set nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    Agent1.Reset(2000, 1.5);
    Agent2.condition = pred_condition;
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    double munch_count = 0;
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    TVector<double> FS;
    TVector<double> PS;
    TVector<double> SS;
    TVector<double> NO1;
    TVector<double> N1FS;
    TVector<double> N1PS;
    TVector<double> N1SS;
    TVector<double> NO2;
    TVector<double> N2FS;
    TVector<double> N2PS;
    TVector<double> N2SS;
    TVector<double> NO3;
    TVector<double> N3FS;
    TVector<double> N3PS;
    TVector<double> N3SS;
    TVector<double> mov;

    // Simulation 1: Food Sense Sample
    for (double time = 0; time < SenseDuration; time += BTStepSize){
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
            double c = rs.UniformRandom(0,1);
            if (c <= BT_G_Rate){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
        }
        TVector<double> pred_pos(0,-1);
        Agent1.Sense(food_pos, pred_pos);
        FS.SetBounds(0, SS.Size());
        FS[FS.Size()-1] = Agent1.f_sensor;
        N1FS.SetBounds(0, N1FS.Size());
        N1FS[N1FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[1];
        N2FS.SetBounds(0, N2FS.Size());
        N2FS[N2FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[4];
        N3FS.SetBounds(0, N3FS.Size());
        N3FS[N3FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[7];
        PS.SetBounds(0, PS.Size());
        PS[PS.Size()-1] = Agent1.p_sensor;
        N1PS.SetBounds(0, N1PS.Size());
        N1PS[N1PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[2];
        N2PS.SetBounds(0, N2PS.Size());
        N2PS[N2PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[5];
        N3PS.SetBounds(0, N3PS.Size());
        N3PS[N3PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[8];
        SS.SetBounds(0, SS.Size());
        SS[SS.Size()-1] = Agent1.state;
        N1SS.SetBounds(0, N1SS.Size());
        N1SS[N1SS.Size()-1] = Agent1.state * Agent1.sensorweights[3];
        N2SS.SetBounds(0, N2SS.Size());
        N2SS[N2SS.Size()-1] = Agent1.state * Agent1.sensorweights[6];
        N3SS.SetBounds(0, N3SS.Size());
        N3SS[N3SS.Size()-1] = Agent1.state * Agent1.sensorweights[9];

        Agent1.Step(BTStepSize, WorldFood);
        NO1.SetBounds(0, NO1.Size());
        NO1[NO1.Size()-1] = Agent1.NervousSystem.NeuronOutput(1);
        NO2.SetBounds(0, NO2.Size());
        NO2[NO2.Size()-1] = Agent1.NervousSystem.NeuronOutput(2);
        NO3.SetBounds(0, NO3.Size());
        NO3[NO3.Size()-1] = Agent1.NervousSystem.NeuronOutput(3);
        mov.SetBounds(0, mov.Size());
        mov[mov.Size()-1] = (Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1));

        if (Agent1.birth == true){
            Agent1.state = Agent1.state - prey_repo;
            Agent1.birth = false;
        }
        if (Agent1.death == true){
            Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
            Agent1.death = false;
        }

        pred_pos.~TVector();
        dead_food.~TVector();
    }
    WorldFood.FillContents(0.0);
    food_pos.~TVector();
    Agent1.Reset(100, 1.5);
    Agent2.Reset(250);

    // Simulation 2: Predator Sense Sample
    for (double time = 0; time < SenseDuration; time += BTStepSize){
        TVector<double> food_pos(0,-1);
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        Agent2.Sense(prey_pos);
        Agent2.Step(BTStepSize, WorldFood, preylist);

        TVector<double> pred_pos(0,-1);
        pred_pos.SetBounds(0, pred_pos.Size());
        pred_pos[pred_pos.Size()-1] = Agent2.pos;

        Agent1.Sense(food_pos, pred_pos);
        FS.SetBounds(0, SS.Size());
        FS[FS.Size()-1] = Agent1.f_sensor;
        N1FS.SetBounds(0, N1FS.Size());
        N1FS[N1FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[1];
        N2FS.SetBounds(0, N2FS.Size());
        N2FS[N2FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[4];
        N3FS.SetBounds(0, N3FS.Size());
        N3FS[N3FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[7];
        PS.SetBounds(0, PS.Size());
        PS[PS.Size()-1] = Agent1.p_sensor;
        N1PS.SetBounds(0, N1PS.Size());
        N1PS[N1PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[2];
        N2PS.SetBounds(0, N2PS.Size());
        N2PS[N2PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[5];
        N3PS.SetBounds(0, N3PS.Size());
        N3PS[N3PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[8];
        SS.SetBounds(0, SS.Size());
        SS[SS.Size()-1] = Agent1.state;
        N1SS.SetBounds(0, N1SS.Size());
        N1SS[N1SS.Size()-1] = Agent1.state * Agent1.sensorweights[3];
        N2SS.SetBounds(0, N2SS.Size());
        N2SS[N2SS.Size()-1] = Agent1.state * Agent1.sensorweights[6];
        N3SS.SetBounds(0, N3SS.Size());
        N3SS[N3SS.Size()-1] = Agent1.state * Agent1.sensorweights[9];

        Agent1.Step(BTStepSize, WorldFood);
        NO1.SetBounds(0, NO1.Size());
        NO1[NO1.Size()-1] = Agent1.NervousSystem.NeuronOutput(1);
        NO2.SetBounds(0, NO2.Size());
        NO2[NO2.Size()-1] = Agent1.NervousSystem.NeuronOutput(2);
        NO3.SetBounds(0, NO3.Size());
        NO3[NO3.Size()-1] = Agent1.NervousSystem.NeuronOutput(3);
        mov.SetBounds(0, mov.Size());
        mov[mov.Size()-1] = (Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1));

        if (Agent1.birth == true){
            Agent1.state = Agent1.state - prey_repo;
            Agent1.birth = false;
        }
        if (Agent1.death == true){
            Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
            Agent1.death = false;
        }
        
        prey_pos.SetBounds(0, prey_pos.Size());
        prey_pos[prey_pos.Size()-1] = Agent1.pos;
        // Pred Sense & Step
        Agent2.Sense(prey_pos);
        Agent2.Step(BTStepSize, WorldFood, preylist);

        prey_pos.~TVector();
        pred_pos.~TVector();
        dead_food.~TVector();
    }

    WorldFood.FillContents(0.0);
    food_pos.~TVector();

    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize));
    // for (int i = 0; i <= CC; i++){
    //     int f = rs.UniformRandomInteger(1,SpaceSize);
    //     WorldFood[f] = 1.0;
    //     food_pos.SetBounds(0, food_pos.Size());
    //     food_pos[food_pos.Size()-1] = f;
    // }

    // Simulation 3: Predator & Food Sense Sample
    for (double time = 0; time < SenseDuration; time += BTStepSize){
        TVector<double> food_pos(0,-1);
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
            double c = rs.UniformRandom(0,1);
            if (c <= BT_G_Rate){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
        }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        Agent2.Sense(prey_pos);
        Agent2.Step(BTStepSize, WorldFood, preylist);

        TVector<double> pred_pos(0,-1);
        pred_pos.SetBounds(0, pred_pos.Size());
        pred_pos[pred_pos.Size()-1] = Agent2.pos;

       Agent1.Sense(food_pos, pred_pos);
        FS.SetBounds(0, SS.Size());
        FS[FS.Size()-1] = Agent1.f_sensor;
        N1FS.SetBounds(0, N1FS.Size());
        N1FS[N1FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[1];
        N2FS.SetBounds(0, N2FS.Size());
        N2FS[N2FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[4];
        N3FS.SetBounds(0, N3FS.Size());
        N3FS[N3FS.Size()-1] = Agent1.f_sensor * Agent1.sensorweights[7];
        PS.SetBounds(0, PS.Size());
        PS[PS.Size()-1] = Agent1.p_sensor;
        N1PS.SetBounds(0, N1PS.Size());
        N1PS[N1PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[2];
        N2PS.SetBounds(0, N2PS.Size());
        N2PS[N2PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[5];
        N3PS.SetBounds(0, N3PS.Size());
        N3PS[N3PS.Size()-1] = Agent1.p_sensor * Agent1.sensorweights[8];
        SS.SetBounds(0, SS.Size());
        SS[SS.Size()-1] = Agent1.state;
        N1SS.SetBounds(0, N1SS.Size());
        N1SS[N1SS.Size()-1] = Agent1.state * Agent1.sensorweights[3];
        N2SS.SetBounds(0, N2SS.Size());
        N2SS[N2SS.Size()-1] = Agent1.state * Agent1.sensorweights[6];
        N3SS.SetBounds(0, N3SS.Size());
        N3SS[N3SS.Size()-1] = Agent1.state * Agent1.sensorweights[9];

        Agent1.Step(BTStepSize, WorldFood);
        NO1.SetBounds(0, NO1.Size());
        NO1[NO1.Size()-1] = Agent1.NervousSystem.NeuronOutput(1);
        NO2.SetBounds(0, NO2.Size());
        NO2[NO2.Size()-1] = Agent1.NervousSystem.NeuronOutput(2);
        NO3.SetBounds(0, NO3.Size());
        NO3[NO3.Size()-1] = Agent1.NervousSystem.NeuronOutput(3);
        mov.SetBounds(0, mov.Size());
        mov[mov.Size()-1] = (Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1));

        if (Agent1.birth == true){
            Agent1.state = Agent1.state - prey_repo;
            Agent1.birth = false;
        }
        if (Agent1.death == true){
            Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
            Agent1.death = false;
        }
        
        prey_pos.SetBounds(0, prey_pos.Size());
        prey_pos[prey_pos.Size()-1] = Agent1.pos;
        // Pred Sense & Step
        Agent2.Sense(prey_pos);
        Agent2.Step(BTStepSize, WorldFood, preylist);

        prey_pos.~TVector();
        pred_pos.~TVector();
        dead_food.~TVector();
    }
    // Save
    SSfile << FS << endl << N1FS << endl << N2FS << endl << N3FS << endl;
    SSfile << PS << endl << N1PS << endl << N2PS << endl << N3PS << endl;
    SSfile << SS << endl << N1SS << endl << N2SS << endl << N3SS << endl; 
    SSfile << NO1 << endl << NO2 << endl << NO3 << endl << mov << endl;

    SSfile.close();
}

// ---------------------------------------
// Eco Fear & Sensory Pollution Functions
// ---------------------------------------
void EcoFear(TVector<double> &genotype, RandomState &rs, double pred_condition)
{
    int maxpred = 30;
    int samplesize = 10;
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey prey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
    // Set nervous system
    prey.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        prey.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        prey.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            prey.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        prey.sensorweights[i] = phenotype(k);
        k++;
    }
    ofstream EFfile("analysis_results/EcoFear.dat");
    for (int j = 0; j<=maxpred; j++)
    {   
        printf("Collecting feeding rates with %d predators\n", j);
        TVector<double> lamb;
        for (int k = 0; k<=samplesize; k++){
            // Fill World to Carrying Capacity
            TVector<double> food_pos;
            TVector<double> WorldFood(1, SpaceSize);
            WorldFood.FillContents(0.0);
            for (int i = 0; i <= CC; i++){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
            // Seed preylist with starting population
            TVector<Predator> predlist(0,-1);
            TVector<Prey> preylist(0,0);
            preylist[0] = prey;
            for (int i = 0; i < j; i++){
                Predator newpred(pred_gain, pred_s_width, pred_frate, pred_handling_time);
                newpred.Reset(rs.UniformRandomInteger(0,SpaceSize));
                newpred.frate = 0.0;
                newpred.condition = pred_condition;
                predlist.SetBounds(0, predlist.Size());
                predlist[predlist.Size()-1] = newpred;
                }

            double munch_count = 0;
            for (double time = 0; time < RateDuration; time += StepSize){
                // Remove chomped food from food list
                TVector<double> dead_food(0,-1);
                for (int i = 0; i < food_pos.Size(); i++){
                    if (WorldFood[food_pos[i]] <= 0){
                        dead_food.SetBounds(0, dead_food.Size());
                        dead_food[dead_food.Size()-1] = food_pos[i];
                    }
                }
                if (dead_food.Size() > 0){
                    for (int i = 0; i < dead_food.Size(); i++){
                        food_pos.RemoveFood(dead_food[i]);
                        food_pos.SetBounds(0, food_pos.Size()-2);
                    }
                }
                // Carrying capacity is 0 indexed, add 1 for true amount
                double food_count = food_pos.Size();
                double s_chance = G_Rate*food_count/(CC+1);
                double c = rs.UniformRandom(0,1);
                if (c > s_chance){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
                // Prey Sense & Step
                TVector<double> pred_pos;
                for (int i = 0; i <= predlist.Size()-1; i++){
                    pred_pos.SetBounds(0, pred_pos.Size());
                    pred_pos[pred_pos.Size()-1] = predlist[i].pos;
                }
                prey.Sense(food_pos, pred_pos);
                prey.Step(StepSize, WorldFood);
                TVector<double> prey_pos(0,0);
                prey_pos[0] = prey.pos;
                // Check Births
                if (prey.birth == true){
                    prey.state = prey.state - prey_repo;
                    prey.birth = false;
                }
                // Check Deaths
                if (prey.death == true){
                    prey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.0);
                    prey.death = false;
                }
                // Check # of times food crossed
                munch_count += prey.snackflag;
                prey.snackflag = 0.0;  
                // Predator Sense & Step
                for(int i = 0; i < predlist.Size(); i++){
                    predlist[i].Sense(prey_pos);
                    predlist[i].Step(StepSize, WorldFood, preylist);
                }
                // Clear lists
                dead_food.~TVector();
                prey_pos.~TVector();
                pred_pos.~TVector();
            }

            double munchrate = munch_count/(RateDuration/StepSize);
            lamb.SetBounds(0, lamb.Size());
            lamb[lamb.Size()-1] = munchrate;
        }
        EFfile << lamb << endl;
        lamb.~TVector();
    }
    // Save
    EFfile.close();
}

void SPoll(TVector<double> &genotype, RandomState &rs, double pred_condition)
{
    // max * interval is greatest sensor width
    const int max_s = 30;
    const double s_interval = 10;
    const int samplesize = 10;
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    Prey prey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_movecost, prey_b_thresh);
    // Set nervous system
    prey.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        prey.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        prey.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            prey.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        prey.sensorweights[i] = phenotype(k);
        k++;
    }
    ofstream SPfile("analysis_results/SPoll.dat");
    for (int j = 0; j<=max_s; j++)
    {   
        prey.s_width = j*s_interval;
        printf("Collecting feeding rates with %f sensor width\n", prey.s_width);
        TVector<double> lamb;
        for (int k = 0; k<=samplesize; k++){
            // Fill World to Carrying Capacity
            TVector<double> food_pos;
            TVector<double> WorldFood(1, SpaceSize);
            WorldFood.FillContents(0.0);
            for (int i = 0; i <= CC; i++){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
            // Seed preylist with starting population
            TVector<Predator> predlist(0,-1);
            TVector<Prey> preylist(0,0);
            preylist[0] = prey;
            for (int i = 0; i <= start_pred; i++){
                Predator newpred(pred_gain, pred_s_width, pred_frate, pred_handling_time);
                newpred.Reset(rs.UniformRandomInteger(0,SpaceSize));
                newpred.frate = 0.0;
                newpred.condition = pred_condition;
                predlist.SetBounds(0, predlist.Size());
                predlist[predlist.Size()-1] = newpred;
                }
            double munch_count = 0;
            for (double time = 0; time < RateDuration; time += StepSize){
                // Remove chomped food from food list
                TVector<double> dead_food(0,-1);
                for (int i = 0; i < food_pos.Size(); i++){
                    if (WorldFood[food_pos[i]] <= 0){
                        dead_food.SetBounds(0, dead_food.Size());
                        dead_food[dead_food.Size()-1] = food_pos[i];
                    }
                }
                if (dead_food.Size() > 0){
                    for (int i = 0; i < dead_food.Size(); i++){
                        food_pos.RemoveFood(dead_food[i]);
                        food_pos.SetBounds(0, food_pos.Size()-2);
                    }
                }
                // Carrying capacity is 0 indexed, add 1 for true amount
                double food_count = food_pos.Size();
                double s_chance = G_Rate*food_count/(CC+1);
                double c = rs.UniformRandom(0,1);
                if (c > s_chance){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
                // Prey Sense & Step
                TVector<double> pred_pos;
                for (int i = 0; i <= predlist.Size()-1; i++){
                    pred_pos.SetBounds(0, pred_pos.Size());
                    pred_pos[pred_pos.Size()-1] = predlist[i].pos;
                }
                prey.Sense(food_pos, pred_pos);
                prey.Step(StepSize, WorldFood);
                TVector<double> prey_pos(0,0);
                prey_pos[0] = prey.pos;
                // Check Births
                if (prey.birth == true){
                    prey.state = prey.state - prey_repo;
                    prey.birth = false;
                }
                // Check Deaths
                if (prey.death == true){
                    prey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.0);
                    prey.death = false;
                }
                // Check # of times food crossed
                munch_count += prey.snackflag;
                prey.snackflag = 0.0;  
                // Predator Sense & Step
                for(int i = 0; i < predlist.Size(); i++){
                    predlist[i].Sense(prey_pos);
                    predlist[i].Step(StepSize, WorldFood, preylist);
                }
                // Clear lists
                dead_food.~TVector();
                prey_pos.~TVector();
                pred_pos.~TVector();
            }

            double munchrate = munch_count/(RateDuration/StepSize);
            lamb.SetBounds(0, lamb.Size());
            lamb[lamb.Size()-1] = munchrate;
        }
        SPfile << lamb << endl;
        lamb.~TVector();
    }
    // Save
    SPfile.close();
}

// ---------------------------------------
// Test Function for Code Development
// ---------------------------------------
void NewEco(TVector<double> &genotype, RandomState &rs)
{
    double test_CC = 29;
    double start_CC = 10;
    double start_prey_sim = 15;
    double test_frate = prey_frate;
    double test_feff = prey_feff;
    ofstream ppfile("menagerie/IndBatch2/analysis_results/ns_15/sim_prey_pop.dat");
    ofstream fpfile("menagerie/IndBatch2/analysis_results/ns_15/sim_food_pop.dat");
    // Translate to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
    // Create agents
    // Playing with feff, frate, metaloss
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
    // Set nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*3; i++) {
        Agent1.sensorweights[i] = phenotype(k);
        k++;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos;
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    for (int i = 0; i <= start_CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        WorldFood[f] = 1.0;
        food_pos.SetBounds(0, food_pos.Size());
        food_pos[food_pos.Size()-1] = f;
    }
    // Make dummy predator list
    TVector<double> pred_pos(0,-1);
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    // // Carrying capacity is 0 indexed, add 1 for true amount
    // for (int i = 0; i < 200; i++){
    //     double food_count = food_pos.Size();
    //     double s_chance = 1 - food_count/(test_CC+1);
    //     double c = rs.UniformRandom(0,1)*50;
    //     if (c < s_chance){
    //         int f = rs.UniformRandomInteger(1,SpaceSize);
    //         WorldFood[f] = 1.0;
    //         food_pos.SetBounds(0, food_pos.Size());
    //         food_pos[food_pos.Size()-1] = f;
    //     }
    // }
    for (int i = 0; i < start_prey_sim; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
        }
    for (double time = 0; time < PlotDuration*300; time += StepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        double c = rs.UniformRandom(0,1);
        for (int i = 0; i < ((test_CC+1) - food_pos.Size()); i++){
            double c = rs.UniformRandom(0,1);
            if (c <= BT_G_Rate){
                int f = rs.UniformRandomInteger(1,SpaceSize);
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        double total_state = 0;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos);
            preylist[i].Step(StepSize, WorldFood);
            total_state += preylist[i].state;
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
                Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_movecost, prey_b_thresh);
                newprey.NervousSystem = preylist[i].NervousSystem;
                newprey.sensorweights = preylist[i].sensorweights;
                newprey.Reset(preylist[i].pos+2, prey_repo);
                newpreylist.SetBounds(0, newpreylist.Size());
                newpreylist[newpreylist.Size()-1] = newprey;
            }
            if (preylist[i].death == true){
                preydeaths.SetBounds(0, preydeaths.Size());
                preydeaths[preydeaths.Size()-1] = i;
            }
        }
        // Update prey list with new prey list and deaths
        if (preydeaths.Size() > 0){
            for (int i = 0; i < preydeaths.Size(); i++){
                preylist.RemoveItem(preydeaths[i]);
                preylist.SetBounds(0, preylist.Size()-2);
            }
        }
        if (newpreylist.Size() > 0){
            for (int i = 0; i < newpreylist.Size(); i++){
                preylist.SetBounds(0, preylist.Size());
                preylist[preylist.Size()-1] = newpreylist[i];
            }
        }
        ppfile << total_state << endl;
        double total_food = 0;
        for (int i = 0; i < WorldFood.Size();i++){
            if (WorldFood[i] > 0){
                total_food += WorldFood[i];
            }
        }
        fpfile << total_food << endl;
        // Check Population Collapse
        if (preylist.Size() <= 0){
            break;
        }
        else{
            newpreylist.~TVector();
            preydeaths.~TVector();
            dead_food.~TVector();
        }
    }
    // Save
    ppfile.close();
    fpfile.close();
}

// ================================================
// E. MAIN FUNCTION
// ================================================
int main (int argc, const char* argv[]) 
{
// ================================================
// EVOLUTION
// ================================================
	// TSearch s(VectSize);
    // long seed = static_cast<long>(time(NULL));
	// // save the seed to a file
	// ofstream seedfile;
	// seedfile.open (seed_string);
	// seedfile << seed << endl;
	// seedfile.close();
	// // Configure the search
	// s.SetRandomSeed(seed);
	// s.SetSearchResultsDisplayFunction(ResultsDisplay);
	// s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	// s.SetSelectionMode(RANK_BASED);
	// s.SetReproductionMode(GENETIC_ALGORITHM);
	// s.SetPopulationSize(POPSIZE);
	// s.SetMaxGenerations(GENS);
	// s.SetCrossoverProbability(CROSSPROB);
	// s.SetCrossoverMode(UNIFORM);
	// s.SetMutationVariance(MUTVAR);
	// s.SetMaxExpectedOffspring(EXPECTED);
	// s.SetElitistFraction(ELITISM);
	// // s.SetSearchConstraint(1);
    // ofstream evolfile;
	// evolfile.open(fitness_string);
	// cout.rdbuf(evolfile.rdbuf());

    // s.SetSearchTerminationFunction(EndTerminationFunction);
    // s.SetEvaluationFunction(Coexist);
    // s.ExecuteSearch();
    // ofstream S1B;
    // TVector<double> bestVector1 = s.BestIndividual();
    // S1B.open(bestgen_string);
    // cout.rdbuf(S1B.rdbuf());
    // S1B << setprecision(32);
    // S1B << bestVector1 << endl;
    // S1B.close();
    // cout.rdbuf(evolfile.rdbuf());
    
    // return 0;

// ================================================
// RUN ANALYSES
// ================================================
    // SET LIST OF AGENTS TO ANALYZE HERE, NEED LIST SIZE FOR INIT
    TVector<int> analylist(0,1);
    analylist.InitializeContents(15);

    // // // Behavioral Traces // // 
    // for (int i = 0; i < analylist.Size(); i++){
    //     int agent = analylist[i];
    //     // load the seed
    //     ifstream seedfile;
    //     double seed;
    //     seedfile.open("seed.dat");
    //     seedfile >> seed;
    //     seedfile.close();
    //     // load best individual
    //     ifstream genefile;
    //     // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    //     genefile.open("menagerie/IndBatch2/genomes/best.gen_16.dat", agent);
    //     TVector<double> genotype(1, VectSize);
    //     genefile >> genotype;
    //     genefile.close();
    //     // set the seed
    //     RandomState rs(seed);
    //     BehavioralTracesCoexist(genotype, rs, pred_condition, agent);
    // }

    // ANALYSES FOR JUST ONE AGENT
    // load the seed
    ifstream seedfile;
    double seed;
    seedfile.open("seed.dat");
    seedfile >> seed;
    seedfile.close();
    // load best individual
    ifstream genefile;
    genefile.open("menagerie/IndBatch2/genomes/best.gen_15.dat");
    TVector<double> genotype(1, VectSize);
	genefile >> genotype;
    genefile.close();
    // set the seed
    RandomState rs(seed);

    // SSCoexist(genotype, rs, pred_condition, 15);

    // // Interaction Rate Collection // //
    CollectEcoRates(genotype, rs);

    // // Sensory Sample Collection // //
    // SensorySample(genotype, rs);

    // // EcoFear Analysis // // 
    // EcoFear(genotype, rs, pred_condition);

    // // Sensory Pollution Analysis // // 
    // SPoll(genotype, rs, pred_condition);

    // // Code Testbed // // 
    // NewEco(genotype, rs);

    return 0;

}