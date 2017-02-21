package opt.test;

import java.text.DecimalFormat;
import java.util.Arrays;

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
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest2 {
    /** The n value */
    private static final int N = 500;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
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
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
        double start = System.nanoTime();
        fit.train();
        double end = System.nanoTime();
        double trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("RHC: " + ";time=" + df1.format(trainingTime) + ";Max = " +  ef.value(rhc.getOptimal()) );

        
        
        double[] CoolingSize = {0.7,0.8,0.95};
        for (int i = 0; i < CoolingSize.length; i++) {
        SimulatedAnnealing sa = new SimulatedAnnealing(100, CoolingSize[i], hcp);
        fit = new FixedIterationTrainer(sa, 20000);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("SA: Cooling = " + CoolingSize[i] + ";time=" + df1.format(trainingTime) + ";Max = " + ef.value(sa.getOptimal()) );
        }

        double[] PopulationSize = {0.5,1,2,4};
        double[] ToMateSize = {0.1,0.2,0.5,0.8,1};
        double[] ToMutationSize = {0.001,0.05,0.2,0.5,0.8,1};
        for (int i = 0; i < PopulationSize.length; i++) {
        	for (int ii = 0; ii < ToMateSize.length; ii++) {
        		for (int iii = 0; iii < ToMutationSize.length; iii++) {
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int) (N*PopulationSize[i]), (int) (N*PopulationSize[i]*ToMateSize[ii]), (int) (N*PopulationSize[i]*ToMutationSize[iii]), gap);
        fit = new FixedIterationTrainer(ga, 300);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("GA: k = " + PopulationSize[i] + ";time=" + df1.format(trainingTime)  + "; ToMate = " + ToMateSize[ii] + "; ToMutation = " + ToMutationSize[iii] + ";Max = " + ef.value(ga.getOptimal()));
        }}}

        double[] MMPopulationSize = {40,60,80,100,120};
        double[] MMPopulationSizeToKept = {0.1,0.2,0.5,0.7,0.9};
        for (int i = 0; i < PopulationSize.length; i++) {
        	for (int ii = 0; ii < MMPopulationSizeToKept.length; ii++) {
        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 100);
        start = System.nanoTime();
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;trainingTime /= Math.pow(10,9);
        System.out.println("MIMIC: k = " + MMPopulationSize[i] + ";time=" + df1.format(trainingTime)  + "; ToKept = " + MMPopulationSizeToKept[ii] + ";Max = " + ef.value(mimic.getOptimal()));
    }}}
}
