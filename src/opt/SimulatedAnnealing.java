package opt;

import dist.Distribution;

import shared.Instance;

/**
 * A simulated annealing hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class SimulatedAnnealing extends OptimizationAlgorithm {
    
    /**
     * The current optimiation data
     */
    private Instance cur;
    
    /**
     * The current optimization value
     */
    private double curVal;
    
    /**
     * The current temperature
     */
    private double t;
    
    /**
     * The initial temperature
     */
    private double t0;
    
    /**
     * The number of iterations to run
     */
    private int iterations;
    
    /**
     * Make a new simulated annealing hill climbing
     * @param t the starting temperature
     * @param iterations the number of iterations to run
     * @param hcp the problem to solve
     */
    public SimulatedAnnealing(double t, int iterations, HillClimbingProblem hcp) {
        super(hcp);
        this.t = t;
        this.t0 = t;
        this.iterations = iterations;
        this.cur = hcp.random();
        this.curVal = hcp.value(cur);
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem p = (HillClimbingProblem) getOptimizationProblem();
        double dt = t0 / iterations;
        while (t > 0) {
            Instance neigh = p.neighbor(cur);
            double neighVal = p.value(neigh);
            double pr = Math.exp((neighVal - curVal) / t);
			if (neighVal > curVal || Distribution.random.nextDouble() < pr) {
                curVal = neighVal;
                cur = neigh;
            }
            t -= dt;
        }
        return curVal;
    }

    /**
     * @see opt.OptimizationAlgorithm#getOptimal()
     */
    public Instance getOptimal() {
        return cur;
    }
    
    public String toString() {
    	return String.format("SimulatedAnnealing(t0=%e,n=%d)", t0, iterations);
    }

}