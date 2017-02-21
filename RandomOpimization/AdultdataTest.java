package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.net.URL;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 * original version from Hannah Lau
 * Adapted to AdultDataset
 * @author Minming Zhao
 * @version 1.0
 */
public class AdultdataTest {
    private static Instance[] instances = initializeInstances();
    private static Instance[] instancesTest = initializeInstancesTest();//Testing set
    
    private static int inputLayer = 63, hiddenLayer = 5, outputLayer = 1, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) { //firstly use HCP and SA
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(2000, 1000, 100, nnop[2]);

        for(int i = 0; i < oa.length-1; i++) { // firstly HCP and SA, but GA take too much time, reduce the iter#
            double start = System.nanoTime(), end, trainingTime, testingTime,testingTime_test, correct = 0, incorrect = 0, correct_test = 0, incorrect_test = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
            
            double predicted_test, actual_test;
            start = System.nanoTime();
            for(int j = 0; j < instancesTest.length; j++) {
                networks[i].setInputValues(instancesTest[j].getData());
                networks[i].run();

                predicted_test = Double.parseDouble(instancesTest[j].getLabel().toString());
                actual_test = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash_test = Math.abs(predicted_test - actual_test) < 0.5 ? correct_test++ : incorrect_test++;

            }
            end = System.nanoTime();
            testingTime_test = end - start;
            testingTime_test /= Math.pow(10,9);
            
            results +=  "\nResults for " + oaNames[i] + ": \nTraining: Correctly classified " + correct + " instances." +
                        "\nTraining: Incorrectly classified " + incorrect + " instances.\nTraining: Percent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining: Training time: " + df.format(trainingTime)
                        + " seconds\nTraining: Testing time: " + df.format(testingTime) + " seconds\nTesting: Correctly classified " + 
                        correct_test + "instances.\nTesting: Incorrectly classified " + incorrect_test + "instances.\nTesting: Percent correctly classified: " + 
                        df.format(correct_test/(correct_test+incorrect_test)*100) + "%\nTest: Test time: " + df.format(testingTime_test) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        //double[][][] attributes = new double[4177][][];
    	double[][][] attributes = new double[25800][][];
    	
        try {
        	//URL url = AbaloneTest.class.getResource("abalone.txt");
        	//URL url = AdultdataTest.class.getResource("adult_data_dummies_clean.csv");
        	URL url = AdultdataTest.class.getResource("adult_data_dummies_train.csv");
        	String strPath = url.getPath();
        	BufferedReader br = new BufferedReader(new FileReader(new File(strPath)));
            //BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/abalone.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[63]; // 63 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 63; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 1; split into 0 and 1
            instances[i].setLabel(new Instance(attributes[i][1][0] < 0.5 ? 0 : 1));
        }

        return instances;
    }


	private static Instance[] initializeInstancesTest() {

    //double[][][] attributes = new double[4177][][];
	double[][][] attributes = new double[6450][][];
	
    try {
    	//URL url = AbaloneTest.class.getResource("abalone.txt");
    	//URL url = AdultdataTest.class.getResource("adult_data_dummies_clean.csv");
    	URL url = AdultdataTest.class.getResource("adult_data_dummies_test.csv");
    	String strPath = url.getPath();
    	BufferedReader br = new BufferedReader(new FileReader(new File(strPath)));
        //BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/abalone.txt")));

        for(int i = 0; i < attributes.length; i++) {
            Scanner scan = new Scanner(br.readLine());
            scan.useDelimiter(",");

            attributes[i] = new double[2][];
            attributes[i][0] = new double[63]; // 63 attributes
            attributes[i][1] = new double[1];

            for(int j = 0; j < 63; j++)
                attributes[i][0][j] = Double.parseDouble(scan.next());

            attributes[i][1][0] = Double.parseDouble(scan.next());
        }
    }
    catch(Exception e) {
        e.printStackTrace();
    }

    Instance[] instances = new Instance[attributes.length];

    for(int i = 0; i < instances.length; i++) {
        instances[i] = new Instance(attributes[i][0]);
        // classifications range from 0 to 1; split into 0 and 1
        instances[i].setLabel(new Instance(attributes[i][1][0] < 0.5 ? 0 : 1));
    }

    return instances;
}
}
