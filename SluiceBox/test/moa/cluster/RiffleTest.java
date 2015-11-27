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

package moa.cluster;

import java.util.ArrayList;

import java.util.TreeMap;

import moa.core.VectorDistances;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.core.Attribute;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsafeUtils;

/**
 *
 * @author bparker
 */
public class RiffleTest {
    
    protected static Instances trainingSet = null;
    protected static Instances testSet = null;
    protected static TreeMap<Integer,Integer> trueVotes = new TreeMap<>();
    /**
     * 2-dim data set array for test use
     * See end of file for R-derived distance tables
     */
    protected static double[][] data = {{0.00, 0.00, 1, 6}, // Data Pt 23 (re-ordered for easier off-by-one lookups to tables
                                        {0.24, 0.24, 1, 1}, // Data Pt 1
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
                                        {0.32, 0.34, 1, 5}}; // Data Pt 22
                                        
    /**
     * test weight options
     */
    protected static double[][] weights = {{0.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0, 1.0, 0.0},{1.0, 1.0, 1.0, 1.0}};
    
    public RiffleTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("Starting test of SubspaceHypersphereClusterWithLabelsTest set");
        
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
               
        for(Attribute a : attribs) {
            UnsafeUtils.setAttributeRange(a, 0.0, 1.0);
        }

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
            x.setClassValue(X[3]);
            x.setWeight(1.0);
            trainingSet.add(x);
            if (trueVotes.containsKey((int) x.classValue())) {
                trueVotes.put((int) x.classValue(), trueVotes.get((int) x.classValue()) + 1);
            } else {
                trueVotes.put((int) x.classValue(), 1);
            }
        }
        
        ArrayList<Attribute> attribs2 = new ArrayList<>(4);
        attribs2.add(new Attribute("x"));
        attribs2.add(new Attribute("y"));
        attribs2.add(new Attribute("z",a3Values));
        attribs2.add(new Attribute("label",labels));
               
        for(Attribute a : attribs2) {
            UnsafeUtils.setAttributeRange(a, 0.0, 1.0);
        }

        attribs2.get(0).setWeight(1.0);
        attribs2.get(1).setWeight(1.0);
        attribs2.get(2).setWeight(0.0);
        attribs2.get(3).setWeight(0.0);
        
        testSet = new Instances("UnitTest-TestSet",attribs2,data.length);
        testSet.setClassIndex(3);
        testSet.setClass(attribs.get(3));
        
        for(double[] X : data) {
            Instance x = new DenseInstance(X.length);
            x.setDataset(testSet);
            for(int i = 0; i < (X.length - 1); ++i) {
                x.setValue(i, X[i]);
            }
            x.setClassValue(X[3]);
            x.setWeight(0.0);
            testSet.add(x);
        }
        assertEquals(data.length, trainingSet.numInstances());
        assertEquals(data.length, testSet.numInstances());
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
     * Test of clear method, of class Riffle.
     */
    @Test
    public void testClear() {
        System.out.println("clear");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        cluster.clear();
        assertEquals(1, cluster.size());
    }

    /**
     * Test of cleanTallies method, of class Riffle.
     */
    @Test
    public void testCleanTallies() {
        System.out.println("cleanTallies");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        cluster.clear();
        int sumOfVotes = (int) weka.core.Utils.sum(cluster.getVotes());
        assertEquals(0.0, sumOfVotes, 0.01);
    }

    /**
     * Test of isEmpty method, of class Riffle.
     */
    @Test
    public void testIsEmpty() {
        System.out.println("isEmpty");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        cluster.addInstance(trainingSet.instance(2));
        assertFalse(cluster.isEmpty());
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        cluster.clear();
        assertTrue(cluster.isEmpty());
    }

    /**
     * Test of isUnlabeled method, of class Riffle.
     */
    @Test
    public void testIsUnlabeled() {
        System.out.println("isUnlabeled");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        cluster.clear();
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        assertTrue(cluster.isUnlabeled());
    }

    /**
     * Test of getNumLabeledPoints method, of class Riffle.
     */
    @Test
    public void testGetNumLabeledPoints() {
        System.out.println("getNumLabeledPoints");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        assertEquals(trainingSet.numInstances(), cluster.getNumLabeledPoints());
    }

    /**
     * Test of size method, of class Riffle.
     */
    @Test
    public void testSize() {
        System.out.println("size");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        assertEquals(trainingSet.numInstances()* 2, cluster.size());
    }

    /**
     * Test of getVotes method, of class Riffle.
     */
    @Test
    public void testGetVotes() {
        System.out.println("getVotes");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        double[] votes = cluster.getVotes();
        for (int i = 0; i < votes.length; ++i) {
            double truth = (trueVotes.containsKey(i)) ? trueVotes.get(i) : 0.0 ;
            assertEquals( truth ,votes[i],0.01);
        }        
    }

    /**
     * Test of getClassProbability method, of class Riffle.
     */
    @Test
    public void testGetClassProbability() {
        System.out.println("getClassProbability");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        double delta = 0.99 / trainingSet.numInstances();
        for(Integer label : trueVotes.keySet()) {
            double truth = (double) trueVotes.get(label) / (double) trainingSet.numInstances();
            double estimate = cluster.getClassProbability(label) ;
            assertEquals(truth , estimate , delta);
        }   
    }

    /**
     * Test of getNumClasses method, of class Riffle.
     */
    @Test
    public void testGetNumClasses() {
        System.out.println("getNumClasses");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            cluster.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            cluster.addInstance(testSet.instance(i));
        }
        assertEquals(trueVotes.size(), cluster.getNumClasses(), 0.5);
    }

        
    /**
     * 
     */
    @Test
    public void testNormalPDF() {
        System.out.println("getNumClasses");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        double delta = 0;
        double computed = 0;
        double target = 0;
        delta = 0;
        target = 0.3989423;
        computed = cluster.normalPDF(delta, 0, 1);
        assertEquals(target, computed, 0.0001);
        delta = 0.25;
        target =  0.3866681;
        computed = cluster.normalPDF(delta, 0, 1);
        assertEquals(target, computed, 0.0001);
        delta = 1;
        target = 0.2419707;
        computed = cluster.normalPDF(delta, 0, 1);
        assertEquals(target, computed, 0.0001);
        delta = 2;
        target = 0.05399097;
        computed = cluster.normalPDF(delta, 0, 1);
        assertEquals(target, computed, 0.0001);
    }
    
    /**
     * Test of getInclusionProbability method, of class Riffle.
     */
    @Test
    public void testGetInclusionProbability() {
        System.out.println("getInclusionProbability");
        int targetIdx = 1;
        double target, computed;
        Instance x;
        
        Riffle universe = new Riffle(trainingSet.instance(targetIdx));
        universe.updateStrategyOption.setChosenIndex(1); // Shephard
        universe.distanceStrategyOption.setChosenIndex(2); // Euclidian
        universe.outlierDefinitionStrategyOption.setChosenIndex(0); // Chavenent Criteria
        Riffle cluster = new Riffle(trainingSet.instance(targetIdx));
        cluster.updateStrategyOption.setChosenIndex(1); // Shephard
        cluster.distanceStrategyOption.setChosenIndex(2); // Euclidian
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0); // Chavenent Criteria
        
        int targetClass = (int) trainingSet.instance(1).classValue();
        for(int i = 0; i < trainingSet.numInstances(); ++i) {
            x = trainingSet.instance(i);
            if (i != targetIdx && x.classValue() == targetClass) { cluster.addInstance(x);}
            universe.addInstance(x);
        }
        
        target = 0.32128;
        x = trainingSet.instance(2);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
        
        target = 0.34076336;
        x = trainingSet.instance(1);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
        
        target = 0.30291582;
        x = trainingSet.instance(3);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
        
        target = 0.2006151276;
        x = trainingSet.instance(21);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
        
        target = 0.0743652;
        x = trainingSet.instance(8);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
        
        target = 0.073748;
        x = trainingSet.instance(18);
        computed = cluster.getInclusionProbability(x);
        assertEquals(target, computed, 0.00001);
    }

    /**
     * Test of isOutlier method, of class Riffle.
     */
    @Test
    public void testIsOutlier() {
        System.out.println("isOutlier");
        int targetIdx = 1;
        Riffle universe = new Riffle(trainingSet.instance(targetIdx));
        universe.updateStrategyOption.setChosenIndex(0); // Stauffer-Grimson
        universe.distanceStrategyOption.setChosenIndex(2); // Euclidian
        universe.outlierDefinitionStrategyOption.setChosenIndex(0); // Chavenent Criteria
        Riffle cluster = new Riffle(trainingSet.instance(targetIdx));
        cluster.updateStrategyOption.setChosenIndex(0); // Stauffer-Grimson
        
        cluster.distanceStrategyOption.setChosenIndex(2); // Euclidian
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0); // Chavenent Criteria
        //cluster.setUniverse(universe);
        cluster.setWeight(0.1);
        int targetClass = (int) trainingSet.instance(targetIdx).classValue();
        for(int j = 0; j < 10; j++) {
            for(int i = 0; i < trainingSet.numInstances(); ++i) {
                Instance x = trainingSet.instance(i);
                universe.addInstance(x);
                if (i != targetIdx && x.classValue() == targetClass) { cluster.addInstance(x);}
            }
        }
        assertFalse(cluster.isOutlier(trainingSet.instance(2)));
        assertFalse(cluster.isOutlier(trainingSet.instance(1)));
        assertFalse(cluster.isOutlier(trainingSet.instance(21)));
        assertTrue(cluster.isOutlier(trainingSet.instance(8)));
        assertTrue(cluster.isOutlier(trainingSet.instance(18)));
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(4); // 1 Sigma
        assertFalse(cluster.isOutlier(trainingSet.instance(2)));
        assertTrue(cluster.isOutlier(trainingSet.instance(1)));
        assertTrue(cluster.isOutlier(trainingSet.instance(21)));
        assertTrue(cluster.isOutlier(trainingSet.instance(8)));
        assertTrue(cluster.isOutlier(trainingSet.instance(18)));
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(1); // 3 Sigma
        assertFalse(cluster.isOutlier(trainingSet.instance(2)));
        assertFalse(cluster.isOutlier(trainingSet.instance(1)));
        assertTrue(cluster.isOutlier(trainingSet.instance(21)));
        assertTrue(cluster.isOutlier(trainingSet.instance(8)));
        assertTrue(cluster.isOutlier(trainingSet.instance(18)));
        
    }

    /**
     * Test of addInstance method, of class Riffle.
     */
    @Test
    public void testAddInstance() {
        System.out.println("addInstance");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            Instance x = trainingSet.instance(i);
            cluster.addInstance(x);
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            Instance x = testSet.instance(i);
            cluster.addInstance(x);
        }
        //assertEquals(trainingSet.numInstances(), cluster.numLabeledPoints);
        assertEquals(trainingSet.numInstances() * 2, cluster.size());
    }

    /**
     * Test of removeInstance method, of class Riffle.
     */
    @Test
    public void testRemoveInstance() {
        System.out.println("removeInstance");
        Riffle cluster = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            Instance x = trainingSet.instance(i);
            cluster.addInstance(x);
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            Instance x = testSet.instance(i);
            cluster.addInstance(x);
        }
        cluster.removeInstance(trainingSet.instance(0));
        //assertEquals(trainingSet.numInstances() - 1, cluster.size());
        assertEquals(trainingSet.numInstances() * 2 - 1, cluster.size());
    }

    /**
     * Test of addInstanceGrimson method, of class Riffle.
     */
    @Test
    public void testAddInstanceGrimson() {
        System.out.println("addInstanceGrimson");
        Riffle cluster = new Riffle(trainingSet.instance(2));
        cluster.updateStrategyOption.setChosenIndex(0);
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0);
        cluster.setWeight(0.1);
        double[] c = cluster.getCenter();
        assertEquals(0.25, c[0], 0.0001);
        assertEquals(0.25, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(1));
        c = cluster.getCenter();
        assertEquals(0.24922, c[0], 0.001);
        assertEquals(0.24922, c[1], 0.001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(3));
        c = cluster.getCenter();
        assertEquals(0.25022753269, c[0], 0.001);
        assertEquals(0.25022753269, c[1], 0.001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(4));
        c = cluster.getCenter();
        assertEquals(0.249494, c[0], 0.001);
        assertEquals(0.250928, c[1], 0.001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(5));
        c = cluster.getCenter();
        assertEquals(0.250278, c[0], 0.001);
        assertEquals(0.250113, c[1], 0.001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
//        double[] ss = cluster.attributeSumSquares;
//        double[] s = cluster.attributeSums;
//        assertEquals(0.0001,Math.sqrt((ss[0] - s[0]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0001,Math.sqrt((ss[1] - s[1]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0000,Math.sqrt((ss[2] - s[2]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0000,Math.sqrt((ss[3] - s[3]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);

           double radius = cluster.getRadius();
           assertEquals(0.0461402, radius, 0.0001);
        
    }

    /**
     * Test of removeInstanceGrimson method, of class Riffle.
     */
    @Test
    public void testRemoveInstanceGrimson() {
        System.out.println("removeInstanceGrimson");
        Riffle cluster = new Riffle(trainingSet.instance(1));
        cluster.updateStrategyOption.setChosenIndex(0);
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0);
        cluster.setWeight(0.1);
        double[] c = cluster.getCenter();
        assertEquals(0.24, c[0], 0.0001);
        assertEquals(0.24, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        cluster.addInstance(trainingSet.instance(6));
        cluster.removeInstance(trainingSet.instance(6));
        c = cluster.getCenter();
        assertEquals(0.24, c[0], 0.0001);
        assertEquals(0.24, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
    }

    /**
     * Test of addInstanceViaShephard method, of class Riffle.
     */
    @Test
    public void testAddInstanceViaShephard() {
        System.out.println("addInstanceViaShephard");
        Riffle cluster = new Riffle(trainingSet.instance(1));
        cluster.updateStrategyOption.setChosenIndex(1);
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0);
        double[] c = cluster.getCenter();
        assertEquals(0.24, c[0], 0.0001);
        assertEquals(0.24, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(2));
        c = cluster.getCenter();
        assertEquals(0.245, c[0], 0.0001);
        assertEquals(0.245, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(3));
        c = cluster.getCenter();
        assertEquals(0.25, c[0], 0.0001);
        assertEquals(0.25, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(4));
        c = cluster.getCenter();
        assertEquals(0.2475, c[0], 0.0001);
        assertEquals(0.2525, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
        cluster.addInstance(trainingSet.instance(5));
        c = cluster.getCenter();
        assertEquals(0.25, c[0], 0.0001);
        assertEquals(0.25, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        
//        double[] ss = cluster.attributeSumSquares;
//        double[] s = cluster.attributeSums;
//        assertEquals(0.0001,Math.sqrt((ss[0] - s[0]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0001,Math.sqrt((ss[1] - s[1]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0000,Math.sqrt((ss[2] - s[2]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);
//        assertEquals(0.0000,Math.sqrt((ss[3] - s[3]*s[1]/cluster.numTotalPoints)) / cluster.numTotalPoints,0.0000001);

           double radius = cluster.getRadius();
           assertEquals(0.05, radius, 0.001);
        
    }

    /**
     * Test of removeInstanceViaShephard method, of class Riffle.
     */
    @Test
    public void testRemoveInstanceViaShephard() {
        System.out.println("removeInstanceViaShephard");
         Riffle cluster = new Riffle(trainingSet.instance(1));
        cluster.updateStrategyOption.setChosenIndex(1);
        
        cluster.outlierDefinitionStrategyOption.setChosenIndex(0);
        double[] c = cluster.getCenter();
        assertEquals(0.24, c[0], 0.0001);
        assertEquals(0.24, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        cluster.addInstance(trainingSet.instance(6));
        cluster.removeInstance(trainingSet.instance(6));
        c = cluster.getCenter();
        assertEquals(0.24, c[0], 0.0001);
        assertEquals(0.24, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
        cluster.addInstance(trainingSet.instance(2));
        cluster.addInstance(trainingSet.instance(3));
        cluster.addInstance(trainingSet.instance(4));
        cluster.addInstance(trainingSet.instance(5));
        cluster.addInstance(trainingSet.instance(7));
        cluster.removeInstance(trainingSet.instance(7));
        c = cluster.getCenter();
        assertEquals(0.25, c[0], 0.0001);
        assertEquals(0.25, c[1], 0.0001);
        assertEquals(1.0,  c[2], 0.0001);
        assertEquals(1.0,  c[3], 0.0001);
    }

    /**
     * Test of getCenterDistance method, of class Riffle.
     */
    @Test
    public void testGetCenterDistance() {
        System.out.println("getCenterDistance");
        Riffle cluster = new Riffle(trainingSet.instance(1));
        
        cluster.distanceStrategyOption.setChosenIndex(2); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        
        assertEquals(0.0, cluster.getCenterDistance(testSet.instance(1)), 0.00000001); 
        assertEquals(0.01414, cluster.getCenterDistance(testSet.instance(2)), 0.00001); 
    }

    /**
     * Test of distanceMinkowski method, of class Riffle.
     */
    @Test
    public void testDistanceMinkowski_Euclidian() {
        System.out.println("distanceMinkowski Euclidian");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Euclidian"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[1], trainingSet, 2.0), 0.00001); 
        assertEquals(Math.sqrt(0.0002), VectorDistances.distanceMinkowski(data[1], data[2], trainingSet, 2.0), 0.00001); // sqrt(0.0002) = 0.01414
        assertEquals(3.01995, VectorDistances.distanceMinkowski(data[2], data[20], trainingSet, 2.0), 0.00001); 
    }
    
    /**
     * Test of distanceMinkowski method, of class Riffle.
     */
    @Test
    public void testDistanceMinkowski() {
        System.out.println("distanceMinkowski p=3");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Euclidian"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[1], trainingSet, 3.0), 0.00001); 
        assertEquals(0.01260, VectorDistances.distanceMinkowski(data[3], data[2], testSet, 3.0), 0.00001); 
        assertEquals(0.02520, VectorDistances.distanceMinkowski(data[4], data[5], testSet, 3.0), 0.00001); 
    }

    /**
     * Test of distanceMinkowski method, of class Riffle.
     */
    @Test
    public void testDistanceMinkowski_Min() {
        System.out.println("distanceMinkowski Minimum");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Minimum"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[1], trainingSet, 0.0), 0.00001); 
        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[2], trainingSet, 0.0), 0.00001); 
        assertEquals(0.01, VectorDistances.distanceMinkowski(data[1], data[2], testSet, 0.0), 0.00001); 
        assertEquals(0.49, VectorDistances.distanceMinkowski(data[0], data[17], testSet, 0.0), 0.00001); 
    }
    
    /**
     * Test of distanceMinkowski method, of class Riffle.
     */
    @Test
    public void testDistanceMinkowski_Manhattan() {
        System.out.println("distanceMinkowski Manhattan");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Manhattan"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[1], trainingSet, 1.0), 0.00001); 
        assertEquals(0.020, VectorDistances.distanceMinkowski(data[1], data[2], testSet, 1.0), 0.00001); 
        assertEquals(0.020, VectorDistances.distanceMinkowski(data[1], data[2], testSet, 1.0), 0.00001); 
        assertEquals(0.020, VectorDistances.distanceMinkowski(data[1], data[2], testSet, 1.0), 0.00001); 
        assertEquals(1.570, VectorDistances.distanceMinkowski(data[5], data[6], trainingSet, 1.0), 0.00001); 
    }
    

    
        /**
     * Test of distanceMinkowski method, of class Riffle.
     */
    @Test
    public void testDistanceMinkowski_Max() {
        System.out.println("distanceMinkowski Maximum/Chebychev");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Chebychev"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceMinkowski(data[1], data[1], trainingSet, 9999999.0), 0.00001); 
        assertEquals(0.51, VectorDistances.distanceMinkowski(data[0], data[17], testSet, 9999999.0), 0.00001); 
        assertEquals(2.0, VectorDistances.distanceMinkowski(data[0], data[17], trainingSet, 9999999.0), 0.00001); 
    }
    
    /**
     * Test of distanceAverage method, of class Riffle.
     */
    @Test
    public void testDistanceAverage() {
        System.out.println("distanceAverage");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Average"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}

        assertEquals(0.0, VectorDistances.distanceAverage(data[1],data[1],trainingSet), 0.00001); 
        assertEquals(Math.sqrt(0.00005), VectorDistances.distanceAverage(data[1],data[2],testSet), 0.00001); //sqrt((0.01^2 + 0.01^2)/4) = 0.00707
        assertEquals(1.01096, VectorDistances.distanceAverage(data[8],data[16],trainingSet), 0.00001); 
    }

    /**
     * Test of distanceGower method, of class Riffle.
     */
    @Test
    public void testDistanceGower() {
        System.out.println("distanceGower (partial)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Gower"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        cluster.updateStrategyOption.setChosenLabel("Stauffer-Grimson");
        cluster.addInstance(trainingSet.instance(0));
        cluster.addInstance(trainingSet.instance(0));
        cluster.addInstance(trainingSet.instance(0));
        
        assertEquals(0.0, VectorDistances.distanceGower(data[1], data[1], trainingSet), 0.001); 
        assertEquals(0.00646, VectorDistances.distanceGower(data[1], data[2], trainingSet), 0.001); 
        assertEquals(0.035, VectorDistances.distanceGower(data[12], data[13], trainingSet), 0.001);
        assertEquals(0.035, VectorDistances.distanceGower(data[13], data[12], trainingSet), 0.001);
    }

    /**
     * Test of distanceDivergence method, of class Riffle.
     */
    @Test
    public void testDistanceDivergence() {
        System.out.println("distanceDivergence (partial)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Divergence"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        
        assertEquals(0.0, VectorDistances.distanceDivergence(data[1], data[1], trainingSet), 0.00001); 
        //assertEquals(0.04082 / 2.0, cluster.distanceDivergence(data[1], data[2], weights[3]), 0.00001); 
        assertEquals(0.11111 / 2.0, VectorDistances.distanceDivergence(data[7], data[6], testSet), 0.00001); 
        assertEquals(0.11111 / 2.0, VectorDistances.distanceDivergence(data[6], data[7], testSet), 0.00001); 
    }

    /**
     * Test of distanceChord method, of class Riffle.
     */
    @Test
    public void testDistanceChord() {
        System.out.println("distanceChord (skipped)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Chord"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
   
        assertEquals(0.0, VectorDistances.distanceChord(data[1], data[1], trainingSet), 0.00001); 
        //assertEquals(0.0, cluster.distanceChord(data[1], data[2], weights[3]), 0.00001); 
    }

    /**
     * Test of distanceGeo method, of class Riffle.
     */
    @Test
    public void testDistanceGeo() {
        System.out.println("distanceGeo (skipped)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Geo"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        
        assertEquals(0.0, VectorDistances.distanceGeo(data[1], data[1], trainingSet), 0.001); 
//        assertEquals(0.0, VectorDistances.distanceGeo(data[1], data[1], trainingSet), 0.00001); //error 0.0 != 0.4879121655698334
    }

    /**
     * Test of distanceGeo method, of class Riffle.
     */
    @Test
    public void testDistanceBray() {
        System.out.println("distanceBray (skipped)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Bray"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        
        assertEquals(0.0, VectorDistances.distanceBray(data[1], data[1], trainingSet), 0.00001); 
        //assertEquals(0.0, cluster.distanceGeo(data[1], data[1], weights[3]), 0.00001); 
    }
    
     /**
     * Test of distanceGeo method, of class Riffle.
     */
    @Test
    public void testDistanceJaccard() {
        System.out.println("distanceJaccard (skipped)");
        Riffle cluster = new Riffle(trainingSet.instance(0).toDoubleArray(),0.10);
        
        cluster.distanceStrategyOption.setChosenLabel("Jaccard"); //{"Minimum", "Manhattan", "Euclidian", "Chebychev", "Average", "Chord", "Geo", "Divergence", "Gower"}
        
        assertEquals(0.0, VectorDistances.distanceJaccard(data[1], data[1], trainingSet), 0.00001); 
        //assertEquals(0.0, cluster.distanceGeo(data[1], data[1], weights[3]), 0.00001); 
    }
    
    /**
     * Test of recompute method, of class Riffle.
     */
    @Test
    public void testRecompute() {
        System.out.println("recompute (skipped)");
        //Not really used
        //fail("The test case is a prototype.");
    }

    /**
     * Test of getTruePurity method, of class Riffle.
     */
    @Test
    public void testGetTruePurity() {
        System.out.println("getTruePurity");
        Riffle instance1 = new Riffle(trainingSet.instance(1));
        for(int i = 0; i < trainingSet.numInstances(); ++i) {
            Instance x = trainingSet.instance(i);
            if (1 != 1 && x.classValue() == trainingSet.instance(1).classValue()) {instance1.addInstance(x);}
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            Instance x = testSet.instance(i);
            if (x.classValue() == trainingSet.instance(1).classValue()) {instance1.addInstance(x);}
        }
        Riffle instance2 = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            instance2.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            instance2.addInstance(testSet.instance(i));
        }
        Riffle instance3 = new Riffle(trainingSet.instance(1));
        for(int i = 2; i < trainingSet.numInstances(); ++i) {
             Instance x = trainingSet.instance(i);
            if (x.classValue() == trainingSet.instance(1).classValue()) {instance3.addInstance(x);}
        }
        for(int i = 0; i < trainingSet.numInstances(); ++i) {
            Instance x = trainingSet.instance(i);
            if (x.classValue() == trainingSet.instance(1).classValue() + 1) {instance3.addInstance(x);}
        }
        assertEquals(1.0, instance1.getPurity(), 1.0 / trainingSet.numInstances());
        assertEquals(1.0 / trueVotes.size(), instance2.getPurity(),  2.0 / trainingSet.numInstances());
        assertEquals(1.0 / 2.0, instance3.getPurity(),  2.0 / trainingSet.numInstances());
    }

    /**
     * Test of getTrueEntropy method, of class Riffle.
     */
    @Test
    public void testGetTrueEntropy() {
        System.out.println("getTrueEntropy (skipped)");
        //pass("Dont care");
    }

    /**
     * Test of getPurity method, of class Riffle.
     */
    @Test
    public void testGetPurity() {
        System.out.println("getPurity");
        Riffle instance1 = new Riffle(trainingSet.instance(1));
        for(int i = 0; i < trainingSet.numInstances(); ++i) {
            Instance x = trainingSet.instance(i);
            if (i != 1 && x.classValue() == trainingSet.instance(1).classValue()) {instance1.addInstance(x);}
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            instance1.addInstance(testSet.instance(i));
        }
        Riffle instance2 = new Riffle(trainingSet.instance(0));
        for(int i = 1; i < trainingSet.numInstances(); ++i) {
            instance2.addInstance(trainingSet.instance(i));
        }
        for(int i = 0; i < testSet.numInstances(); ++i) {
            instance2.addInstance(testSet.instance(i));
        }
        assertEquals(1.0, instance1.getPurity(), 1.0 / trainingSet.numInstances());
        assertEquals(1.0 / trueVotes.size(), instance2.getPurity(),  2.0 / trainingSet.numInstances());
    }

    /**
     * Test of getEntropy method, of class Riffle.
     */
    @Test
    public void testGetEntropy() {
        System.out.println("getEntropy (skipped)");
       
        //pass("Don't care");
    }

    /**
     * Test of safeInit method, of class Riffle.
     */
    @Test
    public void testSafeInit() {
        System.out.println("safeInit");
        Riffle instance1 = new Riffle(trainingSet.instance(0).toDoubleArray(),0.50);
        instance1.addInstance(trainingSet.instance(1));
        assertNotNull(instance1.getVotes());
    }

    /**
     * Test of compareTo method, of class Riffle.
     */
    @Test
    public void testCompareTo() {
        System.out.println("compareTo");
        Riffle cluster1 = new Riffle(trainingSet.instance(0));
        cluster1.addInstance(trainingSet.instance(0));
        Riffle cluster2 = new Riffle(trainingSet.instance(1));
        cluster2.addInstance(trainingSet.instance(0));
        cluster2.addInstance(trainingSet.instance(0));
        Riffle cluster3 = new Riffle(trainingSet.instance(2));
        cluster3.addInstance(trainingSet.instance(0));
        cluster3.addInstance(trainingSet.instance(0));
        cluster3.addInstance(trainingSet.instance(0));
        
        cluster1.setRadius(0.25);
        cluster2.setRadius(0.5);
        cluster3.setRadius(0.5);
        cluster1.setWeight(0.3);
        cluster2.setWeight(0.3);
        cluster3.setWeight(0.6);
        
        assertTrue(cluster1.compareTo(cluster1) == 0);
        assertTrue(cluster2.compareTo(cluster2) == 0);
        assertTrue(cluster1.compareTo(cluster2) > 0);
        assertTrue(cluster2.compareTo(cluster3) > 0);
        assertTrue(cluster3.compareTo(cluster2) < 0);
    }
    
}


    /*
    Distance tables for the above data set for full weights (including label) at 1.0
        Minkowski p=1 (Manhattan)
           1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22
        2  0.020                                                                                                                              
        3  0.040 0.020                                                                                                                        
        4  0.020 0.020 0.020                                                                                                                  
        5  0.020 0.020 0.020 0.040                                                                                                            
        6  1.550 1.550 1.550 1.530 1.570                                                                                                      
        7  1.520 1.500 1.500 1.500 1.520 0.050                                                                                                
        8  1.520 1.500 1.480 1.500 1.500 0.150 0.100                                                                                          
        9  1.600 1.600 1.600 1.580 1.620 0.050 0.100 0.200                                                                                    
        10 1.520 1.500 1.480 1.500 1.500 0.150 0.100 0.000 0.200                                                                              
        11 2.518 2.500 2.500 2.520 2.498 2.050 2.000 1.900 2.100 1.900                                                                        
        12 2.520 2.500 2.500 2.520 2.500 2.050 2.000 1.900 2.100 1.900 0.002                                                                  
        13 2.500 2.500 2.500 2.520 2.480 2.050 2.000 1.900 2.100 1.900 0.098 0.100                                                            
        14 2.500 2.500 2.500 2.520 2.480 2.050 2.000 1.900 2.100 1.900 0.048 0.050 0.050                                                      
        15 2.500 2.500 2.500 2.520 2.480 2.050 2.000 1.900 2.100 1.900 0.032 0.034 0.066 0.016                                                
        16 3.520 3.500 3.480 3.500 3.500 2.570 2.520 2.420 2.620 2.420 1.480 1.480 1.480 1.480 1.480                                          
        17 3.520 3.500 3.480 3.500 3.500 2.530 2.480 2.380 2.580 2.380 1.520 1.520 1.520 1.520 1.520 0.040                                    
        18 3.520 3.500 3.480 3.500 3.500 2.550 2.500 2.400 2.600 2.400 1.500 1.500 1.500 1.500 1.500 0.020 0.020                              
        19 3.530 3.510 3.490 3.510 3.510 2.560 2.510 2.410 2.610 2.410 1.490 1.490 1.490 1.490 1.490 0.010 0.030 0.010                        
        20 3.510 3.490 3.470 3.490 3.490 2.560 2.510 2.410 2.610 2.410 1.490 1.490 1.490 1.490 1.490 0.010 0.030 0.010 0.020                  
        21 4.180 4.160 4.140 4.160 4.160 3.550 3.500 3.400 3.600 3.400 2.500 2.500 2.500 2.500 2.500 1.340 1.340 1.340 1.350 1.330            
        22 4.180 4.160 4.140 4.160 4.160 3.530 3.480 3.380 3.580 3.380 2.520 2.520 2.520 2.520 2.520 1.340 1.340 1.340 1.350 1.330 0.020      
        23 5.480 5.500 5.520 5.500 5.500 4.950 5.000 5.000 5.000 5.000 3.998 4.000 3.900 3.950 3.966 3.000 3.000 3.000 3.010 2.990 1.660 1.660

        Minkowski p=2 (Euclidean)
			1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22
		2  0.01414                                                                                                                                                                        
		3  0.02828 0.01414                                                                                                                                                                
		4  0.02000 0.01414 0.02000                                                                                                                                                        
		5  0.02000 0.01414 0.02000 0.02828                                                                                                                                                
		6  1.12325 1.11915 1.11521 1.11432 1.12414                                                                                                                                        
		7  1.12259 1.11803 1.11364 1.11364 1.12259 0.05000                                                                                                                                
		8  1.10236 1.09772 1.09325 1.09417 1.10145 0.11180 0.07071                                                                                                                        
		9  1.14682 1.14237 1.13807 1.13719 1.14769 0.05000 0.07071 0.14142                                                                                                                
		10 1.10236 1.09772 1.09325 1.09417 1.10145 0.11180 0.07071 0.00000 0.14142                                                                                                        
		11 2.06377 2.06131 2.05894 2.06378 2.05893 1.24595 1.22475 1.18533 1.26689 1.18533                                                                                                
		12 2.06403 2.06155 2.05917 2.06403 2.05917 1.24599 1.22474 1.18533 1.26689 1.18533 0.00141                                                                                        
		13 2.05261 2.05061 2.04871 2.05310 2.04822 1.24599 1.22678 1.18743 1.26886 1.18743 0.06930 0.07071                                                                                
		14 2.05802 2.05578 2.05364 2.05826 2.05340 1.24549 1.22526 1.18585 1.26738 1.18585 0.03394 0.03536 0.03536                                                                        
		15 2.05988 2.05757 2.05535 2.06004 2.05518 1.24554 1.22498 1.18557 1.26711 1.18557 0.02263 0.02404 0.04667 0.01131                                                                
		16 3.02248 3.02079 3.01917 3.02089 3.02076 2.04051 2.03352 2.02193 2.04749 2.02193 1.05603 1.05603 1.05840 1.05662 1.05630                                                        
		17 3.02248 3.02079 3.01917 3.02076 3.02089 2.03512 2.02860 2.01797 2.04162 2.01797 1.06546 1.06546 1.06780 1.06604 1.06573 0.02828                                                
		18 3.02245 3.02076 3.01914 3.02079 3.02079 2.03777 2.03101 2.01990 2.04450 2.01990 1.06066 1.06066 1.06301 1.06125 1.06093 0.01414 0.01414                                        
		19 3.02333 3.02161 3.01995 3.02167 3.02161 2.03926 2.03226 2.02092 2.04600 2.02092 1.05836 1.05835 1.06118 1.05917 1.05878 0.01000 0.02236 0.01000                                
		20 3.02161 3.01995 3.01836 3.02002 3.01995 2.03902 2.03226 2.02092 2.04600 2.02092 1.05834 1.05835 1.06024 1.05870 1.05846 0.01000 0.02236 0.01000 0.01414                        
		21 4.00202 4.00160 4.00122 4.00162 4.00162 3.03205 3.03031 3.02288 3.03937 3.02288 2.04502 2.04519 2.03809 2.04134 2.04250 1.02859 1.02859 1.02849 1.03019 1.02689                
		22 4.00205 4.00162 4.00125 4.00160 4.00170 3.03026 3.02870 3.02159 3.03743 3.02159 2.04752 2.04768 2.04059 2.04383 2.04500 1.02888 1.02849 1.02859 1.03039 1.02708 0.01414        
		23 5.01151 5.01248 5.01350 5.01250 5.01250 4.07462 4.07738 4.07185 4.08412 4.07185 3.10210 3.10242 3.08707 3.09455 3.09703 2.12137 2.12137 2.12132 2.12370 2.11899 1.10354 1.10363
    
        Minkowski p=3
           1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22
        2  0.01260                                                                                                                                                                        
        3  0.02520 0.01260                                                                                                                                                                
        4  0.02000 0.01260 0.02000                                                                                                                                                        
        5  0.02000 0.01260 0.02000 0.02520                                                                                                                                                
        6  1.04241 1.04008 1.03784 1.03779 1.04246                                                                                                                                        
        7  1.04239 1.04004 1.03777 1.03777 1.04239 0.05000                                                                                                                                
        8  1.03151 1.02954 1.02764 1.02769 1.03147 0.10400 0.06300                                                                                                                        
        9  1.05543 1.05268 1.05002 1.04997 1.05548 0.05000 0.06300 0.12599                                                                                                                
        10 1.03151 1.02954 1.02764 1.02769 1.03147 0.10400 0.06300 0.00000 0.12599                                                                                                        
        11 2.01093 2.01030 2.00970 2.01093 2.00970 1.08893 1.07722 1.05739 1.10048 1.05739                                                                                                
        12 2.01099 2.01036 2.00976 2.01099 2.00976 1.08897 1.07722 1.05739 1.10048 1.05739 0.00126                                                                                        
        13 2.00808 2.00758 2.00709 2.00810 2.00708 1.08897 1.07937 1.05940 1.10275 1.05940 0.06174 0.06300                                                                                
        14 2.00946 2.00889 2.00835 2.00947 2.00834 1.08842 1.07776 1.05790 1.10105 1.05790 0.03024 0.03150 0.03150                                                                        
        15 2.00994 2.00935 2.00878 2.00994 2.00878 1.08848 1.07747 1.05763 1.10074 1.05763 0.02016 0.02142 0.04158 0.01008                                                                
        16 3.00131 3.00116 3.00103 3.00118 3.00116 2.00394 2.00293 2.00154 2.00495 2.00154 1.00913 1.00913 1.01031 1.00943 1.00927                                                        
        17 3.00131 3.00116 3.00103 3.00116 3.00118 2.00318 2.00230 2.00114 2.00406 2.00114 1.01158 1.01158 1.01285 1.01190 1.01173 0.02520                                                
        18 3.00130 3.00116 3.00102 3.00116 3.00116 2.00355 2.00260 2.00133 2.00449 2.00133 1.01031 1.01031 1.01153 1.01062 1.01045 0.01260 0.01260                                        
        19 3.00138 3.00123 3.00109 3.00124 3.00123 2.00378 2.00276 2.00144 2.00472 2.00144 1.00973 1.00972 1.01116 1.01014 1.00994 0.01000 0.02080 0.01000                                
        20 3.00123 3.00109 3.00096 3.00110 3.00109 2.00371 2.00276 2.00144 2.00472 2.00144 1.00972 1.00972 1.01068 1.00990 1.00978 0.01000 0.02080 0.01000 0.01260                        
        21 4.00003 4.00002 4.00001 4.00002 4.00002 3.00282 3.00276 3.00188 3.00392 3.00188 2.00616 2.00620 2.00439 2.00522 2.00552 1.00330 1.00330 1.00326 1.00357 1.00299                
        22 4.00003 4.00002 4.00002 4.00002 4.00003 3.00261 3.00256 3.00173 3.00366 3.00173 2.00662 2.00666 2.00479 2.00565 2.00595 1.00340 1.00326 1.00330 1.00364 1.00306 0.01260        
        23 5.00037 5.00042 5.00047 5.00042 5.00042 4.00894 4.00909 4.00769 4.01080 4.00769 3.01605 3.01612 3.01294 3.01447 3.01498 2.02064 2.02064 2.02062 2.02124 2.02002 1.02341 1.02347
    
    
         Minkowski p=Inf (Max)
           1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22
        2  0.010                                                                                                                              
        3  0.020 0.010                                                                                                                        
        4  0.020 0.010 0.020                                                                                                                  
        5  0.020 0.010 0.020 0.020                                                                                                            
        6  1.000 1.000 1.000 1.000 1.000                                                                                                      
        7  1.000 1.000 1.000 1.000 1.000 0.050                                                                                                
        8  1.000 1.000 1.000 1.000 1.000 0.100 0.050                                                                                          
        9  1.000 1.000 1.000 1.000 1.000 0.050 0.050 0.100                                                                                    
        10 1.000 1.000 1.000 1.000 1.000 0.100 0.050 0.000 0.100                                                                              
        11 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000                                                                        
        12 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.001                                                                  
        13 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.049 0.050                                                            
        14 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.024 0.025 0.025                                                      
        15 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.016 0.017 0.033 0.008                                                
        16 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000                                          
        17 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.020                                    
        18 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.010 0.010                              
        19 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.010 0.020 0.010                        
        20 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.010 0.020 0.010 0.010                  
        21 4.000 4.000 4.000 4.000 4.000 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000            
        22 4.000 4.000 4.000 4.000 4.000 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000 1.000 1.000 1.000 0.010      
        23 5.000 5.000 5.000 5.000 5.000 4.000 4.000 4.000 4.000 4.000 3.000 3.000 3.000 3.000 3.000 2.000 2.000 2.000 2.000 2.000 1.000 1.000
    
         Gower
           1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22
        2  0.00646                                                                                                                                                                        
        3  0.01292 0.00646                                                                                                                                                                
        4  0.00625 0.00646 0.00667                                                                                                                                                        
        5  0.00667 0.00646 0.00625 0.01292                                                                                                                                                
        6  0.22271 0.22292 0.22312 0.21646 0.22938                                                                                                                                        
        7  0.21271 0.20625 0.20646 0.20646 0.21271 0.01667                                                                                                                                
        8  0.21375 0.20729 0.20083 0.20750 0.20708 0.04896 0.03229                                                                                                                        
        9  0.23833 0.23854 0.23875 0.23208 0.24500 0.01562 0.03229 0.06458                                                                                                                
        10 0.21375 0.20729 0.20083 0.20750 0.20708 0.04896 0.03229 0.00000 0.06458                                                                                                        
        11 0.27248 0.26665 0.26644 0.27310 0.26581 0.38956 0.37290 0.34060 0.40519 0.34060                                                                                                
        12 0.27312 0.26667 0.26646 0.27312 0.26646 0.38958 0.37292 0.34062 0.40521 0.34062 0.00065                                                                                        
        13 0.26583 0.26562 0.26542 0.27208 0.25917 0.38854 0.37187 0.33958 0.40417 0.33958 0.03165 0.03229                                                                                
        14 0.26635 0.26615 0.26594 0.27260 0.25969 0.38906 0.37240 0.34010 0.40469 0.34010 0.01550 0.01615 0.01615                                                                        
        15 0.26652 0.26631 0.26610 0.27277 0.25985 0.38923 0.37256 0.34027 0.40485 0.34027 0.01033 0.01098 0.02131 0.00517                                                                
        16 0.31812 0.31167 0.30521 0.31188 0.31146 0.28458 0.26792 0.23562 0.30021 0.23562 0.20498 0.20500 0.20396 0.20448 0.20465                                                        
        17 0.31771 0.31125 0.30479 0.31146 0.31104 0.27167 0.25500 0.22271 0.28729 0.22271 0.21790 0.21792 0.21687 0.21740 0.21756 0.01292                                                
        18 0.31792 0.31146 0.30500 0.31167 0.31125 0.27812 0.26146 0.22917 0.29375 0.22917 0.21144 0.21146 0.21042 0.21094 0.21110 0.00646 0.00646                                        
        19 0.32125 0.31479 0.30833 0.31500 0.31458 0.28146 0.26479 0.23250 0.29708 0.23250 0.20810 0.20812 0.20708 0.20760 0.20777 0.00313 0.00979 0.00333                                
        20 0.31479 0.30833 0.30188 0.30854 0.30812 0.28125 0.26458 0.23229 0.29688 0.23229 0.20831 0.20833 0.20729 0.20781 0.20798 0.00333 0.00958 0.00313 0.00646                        
        21 0.25812 0.25167 0.24521 0.25188 0.25146 0.32458 0.30792 0.27562 0.34021 0.27562 0.26498 0.26500 0.26396 0.26448 0.26465 0.16000 0.15958 0.15979 0.16313 0.15667                
        22 0.25792 0.25146 0.24500 0.25167 0.25125 0.31812 0.30146 0.26917 0.33375 0.26917 0.27144 0.27146 0.27042 0.27094 0.27110 0.16021 0.15979 0.16000 0.16333 0.15687 0.00646        
        23 0.40500 0.41146 0.41792 0.41125 0.41167 0.50104 0.51771 0.51875 0.51667 0.51875 0.47748 0.47812 0.44583 0.46198 0.46715 0.42312 0.42271 0.42292 0.42625 0.41979 0.26312 0.26292
    
        Average
           1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22
        2  0.00707                                                                                                                                                                        
        3  0.01414 0.00707                                                                                                                                                                
        4  0.01000 0.00707 0.01000                                                                                                                                                        
        5  0.01000 0.00707 0.01000 0.01414                                                                                                                                                
        6  0.56163 0.55958 0.55761 0.55716 0.56207                                                                                                                                        
        7  0.56129 0.55902 0.55682 0.55682 0.56129 0.02500                                                                                                                                
        8  0.55118 0.54886 0.54663 0.54708 0.55073 0.05590 0.03536                                                                                                                        
        9  0.57341 0.57118 0.56903 0.56859 0.57385 0.02500 0.03536 0.07071                                                                                                                
        10 0.55118 0.54886 0.54663 0.54708 0.55073 0.05590 0.03536 0.00000 0.07071                                                                                                        
        11 1.03189 1.03066 1.02947 1.03189 1.02947 0.62298 0.61237 0.59266 0.63344 0.59266                                                                                                
        12 1.03201 1.03078 1.02959 1.03201 1.02959 0.62300 0.61237 0.59266 0.63344 0.59266 0.00071                                                                                        
        13 1.02630 1.02530 1.02435 1.02655 1.02411 0.62300 0.61339 0.59372 0.63443 0.59372 0.03465 0.03536                                                                                
        14 1.02901 1.02789 1.02682 1.02913 1.02670 0.62275 0.61263 0.59293 0.63369 0.59293 0.01697 0.01768 0.01768                                                                        
        15 1.02994 1.02878 1.02767 1.03002 1.02759 0.62277 0.61249 0.59279 0.63356 0.59279 0.01131 0.01202 0.02333 0.00566                                                                
        16 1.51124 1.51040 1.50959 1.51045 1.51038 1.02026 1.01676 1.01096 1.02374 1.01096 0.52802 0.52802 0.52920 0.52831 0.52815                                                        
        17 1.51124 1.51040 1.50959 1.51038 1.51045 1.01756 1.01430 1.00898 1.02081 1.00898 0.53273 0.53273 0.53390 0.53302 0.53286 0.01414                                                
        18 1.51122 1.51038 1.50957 1.51040 1.51040 1.01888 1.01550 1.00995 1.02225 1.00995 0.53033 0.53033 0.53151 0.53062 0.53047 0.00707 0.00707                                        
        19 1.51166 1.51080 1.50998 1.51084 1.51080 1.01963 1.01613 1.01046 1.02300 1.01046 0.52918 0.52917 0.53059 0.52959 0.52939 0.00500 0.01118 0.00500                                
        20 1.51080 1.50998 1.50918 1.51001 1.50998 1.01951 1.01613 1.01046 1.02300 1.01046 0.52917 0.52917 0.53012 0.52935 0.52923 0.00500 0.01118 0.00500 0.00707                        
        21 2.00101 2.00080 2.00061 2.00081 2.00081 1.51602 1.51516 1.51144 1.51969 1.51144 1.02251 1.02259 1.01904 1.02067 1.02125 0.51430 0.51430 0.51425 0.51510 0.51344                
        22 2.00102 2.00081 2.00062 2.00080 2.00085 1.51513 1.51435 1.51079 1.51872 1.51079 1.02376 1.02384 1.02029 1.02192 1.02250 0.51444 0.51425 0.51430 0.51519 0.51354 0.00707        
        23 2.50575 2.50624 2.50675 2.50625 2.50625 2.03731 2.03869 2.03593 2.04206 2.03593 1.55105 1.55121 1.54353 1.54728 1.54851 1.06068 1.06068 1.06066 1.06185 1.05949 0.55177 0.55182
        
        Divergence (Canberra)
           1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18      19      20      21      22
        2  0.04082                                                                                                                                                                        
        3  0.08000 0.03922                                                                                                                                                                
        4  0.04000 0.04002 0.04000                                                                                                                                                        
        5  0.04000 0.04002 0.04000 0.08000                                                                                                                                                
        6  0.93939 0.94444 0.94892 0.90939 0.97892                                                                                                                                        
        7  0.86889 0.83333 0.83809 0.83889 0.86809 0.11111                                                                                                                                
        8  0.93381 0.89793 0.86310 0.90278 0.89412 0.23448 0.12539                                                                                                                        
        9  0.96270 0.96825 0.97320 0.93368 1.00223 0.03226 0.14337 0.26667                                                                                                                
        10 0.93381 0.89793 0.86310 0.90278 0.89412 0.23448 0.12539 0.00000 0.26667                                                                                                        
        11 1.03307 1.00150 1.00625 1.03627 1.00304 1.28001 1.20100 1.10326 1.30377 1.10326                                                                                                
        12 1.03556 1.00000 1.00476 1.03476 1.00556 1.27895 1.20000 1.10226 1.30276 1.10226 0.00267                                                                                        
        13 1.08027 1.08480 1.08877 1.11980 1.04924 1.33450 1.25263 1.15556 1.35556 1.15556 0.14295 0.14559                                                                                
        14 1.03485 1.03981 1.04425 1.07476 1.00434 1.30603 1.22564 1.12815 1.32854 1.12815 0.06692 0.06958 0.07637                                                                        
        15 1.02148 1.02655 1.03110 1.06145 0.99113 1.29722 1.21729 1.11970 1.32016 1.11970 0.04399 0.04666 0.09924 0.02295                                                                
        16 1.30247 1.26643 1.23134 1.26667 1.26714 0.97963 0.88512 0.76906 1.01026 0.76906 0.65881 0.65766 0.72017 0.68758 0.67773                                                        
        17 1.30247 1.26643 1.23134 1.26714 1.26667 0.94410 0.84813 0.73086 0.97500 0.73086 0.69577 0.69464 0.75595 0.72403 0.71436 0.04000                                                
        18 1.30270 1.26667 1.23158 1.26714 1.26714 0.96190 0.86667 0.75000 0.99267 0.75000 0.67733 0.67619 0.73810 0.70584 0.69608 0.02000 0.02000                                        
        19 1.31135 1.27544 1.24046 1.27579 1.27603 0.96995 0.87544 0.75926 1.00072 0.75926 0.66780 0.66667 0.72845 0.69626 0.68652 0.01010 0.02990 0.00990                                
        20 1.29382 1.25766 1.22246 1.25802 1.25826 0.97158 0.87634 0.75980 1.00221 0.75980 0.66833 0.66718 0.72981 0.69716 0.68729 0.00990 0.03010 0.01010 0.02000                        
        21 0.98246 0.94253 0.90395 0.94321 0.94321 1.06274 0.95539 0.83541 1.08978 0.83541 0.77822 0.77682 0.85451 0.81360 0.80141 0.52052 0.52052 0.52075 0.53022 0.51105                
        22 0.98194 0.94202 0.90345 0.94286 0.94253 1.03549 0.92753 0.80698 1.06285 0.80698 0.80581 0.80441 0.88181 0.84110 0.82895 0.52075 0.52099 0.52110 0.53050 0.51135 0.03031        
        23 2.71429 2.71429 2.71429 2.71429 2.71429 2.50000 2.50000 2.50000 2.50000 2.50000 2.33333 2.33333 2.33333 2.33333 2.33333 2.20000 2.20000 2.20000 2.20000 2.20000 2.09091 2.09091
    
        
    
        */