package opt.test;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest_MMZ {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 120;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        DecimalFormat df1 = new DecimalFormat("0.000");
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        double start = System.nanoTime();
        fit.train();
        double end = System.nanoTime();
        double trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("RHC: "+ ";time=" + df1.format(trainingTime) + ";Max = " + ef.value(rhc.getOptimal()));
        
        double[] CoolingSize = {0.7,0.8,0.95};
        for (int i = 0; i < CoolingSize.length; i++) {
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        //fit_its = fit.getIterations();
        System.out.println("SA: Cooling = " + CoolingSize[i] + ";time=" + df1.format(trainingTime) + ";Max = " + ef.value(sa.getOptimal()) );
        }
        
        double[] PopulationSize = {0.5,1,2,4};
        double[] ToMateSize = {0.1,0.2,0.5,0.8,1};
        double[] ToMutationSize = {0.001,0.05,0.2,0.5,0.8,1};
        for (int i = 0; i < PopulationSize.length; i++) {
        	for (int ii = 0; ii < ToMateSize.length; ii++) {
        		for (int iii = 0; iii < ToMutationSize.length; iii++) {
        // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int) (NUM_ITEMS*PopulationSize[i]), (int) (NUM_ITEMS*PopulationSize[i]*ToMateSize[ii]), (int) (NUM_ITEMS*PopulationSize[i]*ToMutationSize[iii]), gap);
        fit = new FixedIterationTrainer(ga, 1000);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("GA: k = " + PopulationSize[i] + ";time=" + df1.format(trainingTime)  + "; ToMate = " + ToMateSize[ii] + "; ToMutation = " + ToMutationSize[iii] + ";Max = " + ef.value(ga.getOptimal()));
        }}}
        
        double[] MMPopulationSize = {40,60,80,100,120,140,200};
        double[] MMPopulationSizeToKept = {0.1,0.2,0.5,0.7,0.9};
        for (int i = 0; i < MMPopulationSize.length; i++) {
        	for (int ii = 0; ii < MMPopulationSizeToKept.length; ii++) {
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("MIMIC: k = " + MMPopulationSize[i] + ";time=" + df1.format(trainingTime)  + "; ToKept = " + MMPopulationSizeToKept[ii] + ";Max = " + ef.value(mimic.getOptimal()));
        }}
    }

}
