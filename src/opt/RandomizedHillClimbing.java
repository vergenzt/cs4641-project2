package opt;

import shared.Instance;

/**
 * A randomized hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class RandomizedHillClimbing extends OptimizationAlgorithm {
    
    /**
     * The number of times to restart
     */
    private int numRestarts;
    
    /**
     * The number of lower-fitness neighbors to check before
     * deciding we're at the top of a hill
     */
    private int numPlateauChecks;
    
    /**
     * The current optimization data
     */
    private Instance cur;
    
    /**
     * The current value of the data
     */
    private double curVal;
    
    /**
     * Make a new randomized hill climbing
     */
    public RandomizedHillClimbing(HillClimbingProblem hcp) {
    	this(0, 5, hcp);
    }
    
    /**
     * Make a new randomized hill climbing with random restarts and a number of plateau checks.
     */
    public RandomizedHillClimbing(int numRestarts, int numPlateauChecks, HillClimbingProblem hcp) {
        super(hcp);
        this.numRestarts = numRestarts;
        this.numPlateauChecks = numPlateauChecks;
        cur = hcp.random();
        curVal = hcp.value(cur);
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        Instance bestCur = null;
        double bestCurVal = Double.NEGATIVE_INFINITY;
        for (int i=0; i < numRestarts + 1; i++) {
        	int numPlateauNeighborsChecked = 0;
        	while (numPlateauNeighborsChecked < numPlateauChecks) {
	            Instance neigh = hcp.neighbor(cur);
	            double neighVal = hcp.value(neigh);
	            if (neighVal > curVal) {
	            	numPlateauNeighborsChecked = 0;
	                curVal = neighVal;
	                cur = neigh;
	            }
	            else {
	            	numPlateauNeighborsChecked++;
	            }
        	}
        	if (curVal > bestCurVal) {
        		bestCur = cur;
        		bestCurVal = curVal;
        	}
        }
        cur = bestCur;
        curVal = bestCurVal;
        return curVal;
    }

    /**
     * @see opt.OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        return cur;
    }

    public String toString() {
    	return String.format("RandomizedHillClimbing(restarts=%d,numPlateauChecks=%d)", numRestarts, numPlateauChecks);
    }
}
