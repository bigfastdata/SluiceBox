/*
 * Copyright 2014 bparker.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package moa.clusterer;

import java.util.ArrayList;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsafeUtils;

/**
 *
 * @author bparker
 */
public class FeS2Test {
    
    protected static Instances trainingSet = null;
    protected static Instances testSet = null;
    protected static double[] trueVotes = null;
    protected static double[][] data = {{0.24, 0.24, 1, 1}, // Data Pt 1
                                        {0.25, 0.25, 1, 1}, // Data Pt 2
                                        {0.26, 0.26, 1, 1}, // Data Pt 3
                                        {0.24, 0.26, 1, 1}, // Data Pt 4
                                        {0.26, 0.24, 1, 1}, // Data Pt 5
                                        {0.20, 0.75, 1, 2}, // Data Pt 6
                                        {0.25, 0.75, 1, 2}, // Data Pt 7
                                        {0.30, 0.70, 1, 2}, // Data Pt 8
                                        {0.20, 0.80, 1, 2}, // Data Pt 9
                                        {0.30, 0.70, 1, 2}, // Data Pt 10
                                        {0.749, 0.249, 1, 3}, // Data Pt 11
                                        {0.750, 0.250, 1, 3}, // Data Pt 12
                                        {0.700, 0.200, 1, 3}, // Data Pt 13
                                        {0.725, 0.225, 1, 3}, // Data Pt 14
                                        {0.733, 0.233, 1, 3}, // Data Pt 15
                                        {0.51, 0.49, 1, 4}, // Data Pt 16
                                        {0.49, 0.51, 1, 4}, // Data Pt 17
                                        {0.50, 0.50, 1, 4}, // Data Pt 18
                                        {0.51, 0.50, 1, 4}, // Data Pt 19
                                        {0.50, 0.49, 1, 4}, // Data Pt 20
                                        {0.33, 0.33, 1, 5}, // Data Pt 21
                                        {0.32, 0.34, 1, 5}, // Data Pt 22
                                        {0.00, 0.00, 1, 6}}; // Data Pt 23
    
   
    public FeS2Test() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("Starting test of FeS2Test set");
        ArrayList<String> a3Values = new ArrayList<>(3);
        a3Values.add("0");
        a3Values.add("1");
        a3Values.add("2");
        a3Values.add("3");
        a3Values.add("4");
        a3Values.add("5");
        a3Values.add("6");
        a3Values.add("7");
        a3Values.add("8");
        a3Values.add("9");
        
        ArrayList<String> labels = new ArrayList<>(3);
        labels.add("Class0");
        labels.add("Class1");
        labels.add("Class2");
        labels.add("Class3");
        labels.add("Class4");
        labels.add("Class5");
        labels.add("Class6");
        labels.add("Class7");
        labels.add("Class8");
        labels.add("Class9");
        
        ArrayList<Attribute> attribs = new ArrayList<>(4);
        attribs.add(new Attribute("x"));
        attribs.add(new Attribute("y"));
        attribs.add(new Attribute("z",a3Values));
        attribs.add(new Attribute("label",labels));
           
        UnsafeUtils.setAttributeRange(attribs.get(0), 0.0, 1.0);
        UnsafeUtils.setAttributeRange(attribs.get(1), 0.0, 1.0);

        attribs.get(0).setWeight(1.0);
        attribs.get(1).setWeight(1.0);
        attribs.get(2).setWeight(1.0);
        attribs.get(3).setWeight(1.0);
        
        trainingSet = new Instances("UnitTest-TrainingSet",attribs,data.length);
        trainingSet.setClassIndex(3); //zero-indexed 4th
        trainingSet.setClass(attribs.get(3)); //zero-indexed 4th
        
        for(double[] X : data) {
            Instance x = new DenseInstance(X.length);
            x.setDataset(trainingSet);
            for(int i = 0; i < (X.length - 1); ++i) {
                x.setValue(i, X[i]);
            }
            x.setClassValue(labels.get((int) X[X.length - 1]));
            x.setWeight(1.0);
            trainingSet.add(x);
        }
        
        testSet = new Instances("UnitTest-TestSet",attribs,data.length);
        testSet.setClassIndex(3);
        testSet.setClass(attribs.get(3));
        
        for(double[] X : data) {
            Instance x = new DenseInstance(X.length);
            x.setDataset(testSet);
            for(int i = 0; i < (X.length - 1); ++i) {
                x.setValue(i, X[i]);
            }
            x.setClassValue(labels.get((int) X[X.length - 1]));
            x.setWeight(0.0);
            testSet.add(x);
        }
    }
    
    @AfterClass
    public static void tearDownClass() {
        trainingSet.clear();
        testSet.clear();
        System.out.println("Test set complete.");
    }
    
    @Before
    public void setUp() {
         
    }
    
    @After
    public void tearDown() {

    }

    /**
     * Test of resetLearningImpl method, of class FeS2.
     */
    @Test
    public void testResetLearningImpl() {
        System.out.println("resetLearningImpl");
        FeS2 instance = new FeS2();
        instance.resetLearningImpl();
        assertEquals(instance.getClusteringResult().size(),0);
    }

    /**
     * Test of trainOnInstance method, of class FeS2.
     */
    @Test
    public void testTrainOnInstance() {
        System.out.println("trainOnInstance");
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0); // none
        classifier.distanceStrategyOption.setChosenIndex(2); // Euclidean
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0); // Chauvenet
        //classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(0); // 0 = Grimson, 1= Shephard
        classifier.initialStandardDeviationOption.setValue(0.5);
        double weightSum = 0;
        
        // Train on 1st instance of first class
        classifier.trainOnInstance(trainingSet.instance(0));
        weightSum += trainingSet.instance(0).weight();
        assertEquals(1, classifier.getClusteringResult().size());
        assertEquals(weightSum, classifier.trainingWeightSeenByModel(),0.001);
        
        // Train on remaining 1st class
        for(Instance x : trainingSet) {
            if(x.classValue() == 1) {
                if (x != trainingSet.instance(0)) {
                    classifier.trainOnInstance(x);
                    weightSum += x.weight();
                }
            }
        }
        assertEquals(weightSum, classifier.trainingWeightSeenByModel(),0.001);
        assertEquals(1, classifier.getClusteringResult().size());
        
        // Train on 2rd class
        for(Instance x : trainingSet) {
            if(x.classValue() == 2) {
                classifier.trainOnInstance(x);
                weightSum += x.weight();
            }
        }
        assertEquals(weightSum, classifier.trainingWeightSeenByModel(),0.001);
        assertEquals(2, classifier.getClusteringResult().size());
        
        // Train on 3rd class
        for(Instance x : trainingSet) {
            if(x.classValue() == 3) {
                classifier.trainOnInstance(x);
                weightSum += x.weight();
            }
        }
        assertEquals(weightSum, classifier.trainingWeightSeenByModel(),0.001);
        assertEquals(3, classifier.getClusteringResult().size());
    }

    /**
     * Test of trainOnInstanceImpl method, of class FeS2.
     */
    @Test
    public void testTrainOnInstanceImpl() {
        System.out.println("trainOnInstanceImpl (skipped)");
        
    }

    /**
     * Test of getVotesForInstance method, of class FeS2.
     */
    @Test
    public void testGetVotesForInstance() {
        System.out.println("getVotesForInstance");
        
        FeS2 classifier = new FeS2();
        Instance x = trainingSet.instance(0);
        classifier.subspaceStrategyOption.setChosenIndex(0);
        classifier.distanceStrategyOption.setChosenIndex(2);
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0);
       // classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(1);
        classifier.trainOnInstance(trainingSet.instance(0));
        double[] result = classifier.getVotesForInstance(x);
        int h = (int) result[weka.core.Utils.maxIndex(result)];
        int y = 1;
        assertEquals(y,h);
        
        // TODO - add fuller set
    }

    /**
     * Test of getModelMeasurementsImpl method, of class FeS2.
     */
    @Test
    public void testGetModelMeasurementsImpl() {
        System.out.println("getModelMeasurementsImpl (skipped)");
    }

    /**
     * Test of getAverageTruePurity method, of class FeS2.
     */
    @Test
    public void testGetAverageTruePurity() {
        System.out.println("getAverageTruePurity");
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0);
        classifier.distanceStrategyOption.setChosenIndex(2);
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0);
       // classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(1);
        double expResult = 1.0;
        for(Instance x : trainingSet) {
                 classifier.trainOnInstance(x);
        }
        for(Instance x : testSet) {
                 classifier.trainOnInstance(x);
        }
        double result = classifier.getAverageTruePurity();
        assertEquals(expResult, result, 0.00001);
    }

    /**
     * Test of getAverageTrueEntropy method, of class FeS2.
     */
    @Test
    public void testGetAverageTrueEntropy() {
        System.out.println("getAverageTrueEntropy (skipped)");
    }

    /**
     * Test of getAveragePurity method, of class FeS2.
     */
    @Test
    public void testGetAveragePurity() {
        System.out.println("getAveragePurity");
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0);
        classifier.distanceStrategyOption.setChosenIndex(2);
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0);
     //  classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(1);
        double expResult = 1.0;
        for(Instance x : trainingSet) {
                 classifier.trainOnInstance(x);
         }
        double result = classifier.getAverageTruePurity();
        assertEquals(expResult, result, 0.00001);
    }

    /**
     * Test of getAverageEntropy method, of class FeS2.
     */
    @Test
    public void testGetAverageEntropy() {
        System.out.println("getAverageEntropy (skipped)");
    }

    /**
     * Test of getAverageVariance method, of class FeS2.
     */
    @Test
    public void testGetAverageVariance() {
        System.out.println("getAverageVariance (skipped)");
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0);
        classifier.distanceStrategyOption.setChosenIndex(2);
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0);
      //  classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(1);
//        double expResult = 0.667 / trainingSet.numClasses();
//        for(Instance x : trainingSet) {
//                 classifier.trainOnInstance(x);
//         }
//        double result = classifier.getAverageVariance();
//        assertEquals(expResult, result, 0.005);
    }

    /**
     * Test of getAverageSize method, of class FeS2.
     */
    @Test
    public void testGetAverageSize() {
        System.out.println("getAverageSize");
        
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0); // none
        classifier.distanceStrategyOption.setChosenIndex(2); // Euclidean
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0); // Chauvenet
      //  classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(0); // 0 = Grimson-Stauffer, 1= Shephard
        classifier.initialStandardDeviationOption.setValue(0.15);
        double expResult = trainingSet.numInstances() /  6.0 ;
        for(Instance x : trainingSet) {
                 classifier.trainOnInstance(x);
         }
        double result = classifier.getAverageSize();
        assertEquals(expResult, result, 0.005);
    }

    /**
     * Test of getAverageLabels method, of class FeS2.
     */
    @Test
    public void testGetAverageLabels() {
        System.out.println("getAverageLabels");
        
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0); // none
        classifier.distanceStrategyOption.setChosenIndex(2); // Euclidean
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0); // Chauvenet
       // classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(0); // 0 = Grimson-Stauffer, 1= Shephard
        classifier.initialStandardDeviationOption.setValue(0.15);
        double expResult = trainingSet.numInstances() /  6.0 ;
        for(Instance x : trainingSet) {
                 classifier.trainOnInstance(x);
        }
        for(Instance x : testSet) {
                 classifier.trainOnInstance(x);
        }
        double result = classifier.getAverageLabels();
        assertEquals(expResult, result, 0.005);
        
    }

    /**
     * Test of getModelDescription method, of class FeS2.
     */
    @Test
    public void testGetModelDescription() {
        System.out.println("getModelDescription (skipped)");
    }

    /**
     * Test of isRandomizable method, of class FeS2.
     */
    @Test
    public void testIsRandomizable() {
        System.out.println("isRandomizable");
        FeS2 instance = new FeS2();
        boolean expResult = false;
        boolean result = instance.isRandomizable();
        assertEquals(expResult, result);
    }

    /**
     * Test of getClusteringResult method, of class FeS2.
     */
    @Test
    public void testGetClusteringResult() {
        System.out.println("getClusteringResult");
        FeS2 classifier = new FeS2();
        classifier.subspaceStrategyOption.setChosenIndex(0); // none
        classifier.distanceStrategyOption.setChosenIndex(2); // Euclidean
        classifier.initialClusterWeightOption.setValue(0.1);
        classifier.learningRateAlphaOption.setValue(0.95);
        classifier.minimumClusterSizeOption.setValue(3);
        classifier.outlierDefinitionStrategyOption.setChosenIndex(0); // Chauvenet
       // classifier.pruneThresholdOption.setValue(0.00001);
        classifier.updateStrategyOption.setChosenIndex(0); // 0 = Grimson-Stauffer, 1= Shephard
        classifier.initialStandardDeviationOption.setValue(0.15);
         
        for(Instance x : trainingSet) {
            classifier.trainOnInstance(x);
        }
        assertEquals(6, classifier.getClusteringResult().size());
    }
    
}
