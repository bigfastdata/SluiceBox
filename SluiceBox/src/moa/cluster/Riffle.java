/*
 *    Riffle.java
 *    Copyright (C) 2013 Brandon S. Parker
 *    @author Brandon S. Parker (brandon.parker@utdallas.edu)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 *    
 */
package moa.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import moa.classifiers.Classifier;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.clusterer.outliers.Sieve;
import moa.core.VectorDistances;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.MultiChoiceOption;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Riffle.java This class was originally designed for use by the SluiceBox for MOA as part of
 * Brandon Parker's Dissertation work. We are finding subspace clusters. A subspace is within the n-dim full feature space
 * of the given data.
 *
 * 
 * RIFFLE - Relative Induction Frame For Latent (Label) Estimation
 * (alternatively RIFFLE - Running Incremental Frame For Latent (Label) Estimation)
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
final public class Riffle extends SphereCluster implements Comparable<SphereCluster> {

    private static final long serialVersionUID = 1L;

    public final MultiChoiceOption updateStrategyOption = new MultiChoiceOption("UpdateStrategy", 'u',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Stauffer-Grimson", "Shephard", "cache"},
            new String[]{"Gaussian update based on Stauffer and Grimson (1999)", "Robust update to momemts based on Shephard (1996)", "cache and compute"},
            2);

    public final MultiChoiceOption outlierDefinitionStrategyOption = new MultiChoiceOption("outlierDefinitionStrategy", 'o',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Chauvenet", "Learned", "2.5-sigma", "3-sigma", "6-sigma","oracle"},
            new String[]{"Chauvenet", "Used trained Perceptron", "2.5-sigma", "3-sigma", "6-sigma","Cheat and use ground truth (for unit testing purposes)"},
            0);

    public final MultiChoiceOption distanceStrategyOption = (MultiChoiceOption) VectorDistances.distanceStrategyOption.copy();

    public final FloatOption initialStandardDeviationOption = new FloatOption("initialStandardDeviation", 's',
            "intial StdDev for cluster upon intial creation (first data point)",
            0.05, 0.0, 10000.0);

    public final FloatOption alphaAdjustmentWeightOption = new FloatOption("clusterAdjustmentWeight", 'a',
            "akin to alpha value in Stauffer & Grimson (1999). How much a new instance affects a cluster centroid position. Only valid if option u=0",
            0.05, 0.0, 0.5);
    
    public ClassOption embeddedLearnerOption = new ClassOption("embeddedLearner", 'E',
            "Classifier for cluster label hypothesis", Classifier.class, "functions.MajorityClass");
    /**
     * Magic numbers
     */
    public final static double GAUSSIAN_DIVISOR = Math.sqrt(2.0 * Math.PI);
    public final static double LAPLACIAN_SMOOTHING_CONSTANT = 1e-32;
    public final static double PI2E = 17.07946844534713193298; // 2 * PI * e
    public final static double LOG2 = Math.log(2); // Used for entropy (Bits) - otherwise ln(x) is in "Nits"
    
    final static AtomicInteger autoindex = new AtomicInteger(0);
    private int numLabeledPoints = 0;
    private int numTotalPoints = 0;
    private double[] centroid = null;
    private double[] labelFrequencies = null;
    private double[] gtLabelFrequencies = null;
    private double[][] symbolFrequencies = null;
    private double[] variances = null;
    private double[] entropies = null;
    private double[] rho = null;
    public Instances instances = null;
    // Used for computing variance (axis-parallel or diagonal covariance) from the centriod (avoiding full covariance computations)
    private double runningSumOfSquares = 0;
    private Sieve parentClusterer = null;
    private Classifier embeddedClassifier = null;
    boolean excludeOutlierVoting = false;
   
    /**
     * Specialized CCTor to copy from base class
     * @param c 
     */
    public Riffle(SphereCluster c) {
        numTotalPoints = 1;
        runningSumOfSquares = 0;
        this.setCenter(c.getCenter()); // For parent compatibility
        this.centroid = this.getCenter();
        this.setRadius(c.getRadius());
        Arrays.fill(this.variances, c.getRadius());
        this.setWeight(c.getWeight());
        this.setId(autoindex.getAndIncrement()); 
    }
    
    /**
     * Create a new cluster from an exemplar data point
     * @param x 
     */
    public Riffle(Instance x) {
        safeInit(x);        
        this.numLabeledPoints = (int) Math.ceil(x.weight());
        this.labelFrequencies[(int) x.classValue()] += x.weight();
        this.gtLabelFrequencies[(int) x.classValue()]++;
        for (int i = 0; (i < this.symbolFrequencies.length) && (i < x.numAttributes()); ++i) {
            double value = x.value(i);
            if (this.symbolFrequencies[i] == null) {
                if ((this.parentClusterer != null) && (this.parentClusterer.getUniverse() != null )){
                    this.variances[i] = this.parentClusterer.getUniverse().variances[i];
                } else {
                    this.variances[i] = this.initialStandardDeviationOption.getValue();
                }
            } else {
                this.variances[i] = 1;
                this.symbolFrequencies[i][(int)value]++;
            }
        }
        this.numTotalPoints = 1;
        this.setGroundTruth(x.classValue());
        this.setCenter(x.toDoubleArray());        
        this.setWeight(x.weight());
        this.setRadius(this.initialStandardDeviationOption.getValue());    
        this.runningSumOfSquares = 0.0;
        this.setId(autoindex.getAndIncrement()); 
    }

    /**
     * Create a new cluster from a pre-computed centroid
     * @param centroidArg new cluster center
     * @param radiusArg
     */
    public Riffle(double[] centroidArg, double radiusArg) {
        numTotalPoints = 1;
        runningSumOfSquares = 0;
        this.setCenter(centroidArg);
        this.centroid = this.getCenter();
        this.setRadius(radiusArg);
        this.setId(autoindex.getAndIncrement()); 
    }

    /**
     * 
     */
    public void prepareEmbeddedClassifier() {
        embeddedClassifier = (Classifier) this.embeddedLearnerOption.materializeObject(null, null);
        embeddedClassifier.prepareForUse();
        embeddedClassifier.resetLearning();
    }
    
    /**
     * Clear out all counters, centroids, etc
     * Note that size will be set to 1 since it will still have a center point 'exemplar'
     */
    public void clear() {
        numLabeledPoints = 0;
        numTotalPoints = 0;
        runningSumOfSquares = 0;
        //this.setId(autoindex.getAndIncrement()); // is this necessary? A: no, and it will mess with TreeSets!
        this.setGroundTruth(-1);
        this.setRadius(weka.core.Utils.SMALL); // we don't want this zero, as it is used as a denominator in many places
        this.setWeight(0);
        Arrays.fill(centroid, 0.0);
        this.setCenter(centroid);
        cleanTallies();
        this.embeddedClassifier.resetLearning();
    }

    /**
     * Optional setter function to give knowledge of the parent clusterer from which to extract p(x) and other info
     * @param o FeS2 object
     */
    public final void setParentClusterer(Sieve o) {
        this.parentClusterer = o;
    }
    
    /**
     * Return current online approximation of variance per attribute
     * Note that this is a vector, not a covariance matrix, so one can assume that this is an axis-parallel (i.e. diagonal)
     * approximation of the covariance assuming attribute independence. 
     * @return array of attribute variances
     */
    public final double[] getVariances() {
        return this.variances;
    }
    
     /**
     * Return current online approximation of entropy per attribute
     * @return array of attribute entropies
     */
    public final double[] getEntropies() {
        return this.entropies;
    }
    
    public final void setVariances(double[] vars) {
        this.variances = new double[vars.length];
        for(int i = 0; i < vars.length; ++i)
        {
            this.variances[i] = vars[i];
        }
    }
    
    public final Instances getHeader() { 
        return this.instances; 
    }
    
    /**
     * 
     * @param W array of attribute weights
     */
    public void setAttributeWeights(double[] W) {
        for(int i = 0; i < W.length && i < this.instances.numAttributes(); ++i)
        {
            this.instances.attribute(i).setWeight(W[i]);
        }
    }
    
    public final void resetEmbeddedLearning() {
        this.embeddedClassifier.resetLearning();
    }
    
    /**
     * 
     * @param x Training Instance
     */
    public void trainEmbeddedClassifier(Instance x) {
        this.embeddedClassifier.trainOnInstance(x);
    }
    
    /**
     * Only clean out label and variance trackers
     */
    public final void cleanTallies() {
        if (this.labelFrequencies != null) {
            Arrays.fill(labelFrequencies, 0);
        }
        if (this.gtLabelFrequencies != null) {
            Arrays.fill(gtLabelFrequencies, 0);
        }
        this.numTotalPoints = 1;
        this.numLabeledPoints = 0;
        this.embeddedClassifier.resetLearning();
    }

    /**
     * chck if the cluster contains no points
     * @return true if cluster was not provided any points
     */
    public final boolean isEmpty() {
        return (this.numTotalPoints <= 1);
    }

    /**
     * Checks if cluster has received any labeled training data.
     * @return  true if no labeled data points are in this cluster
     */
    public final boolean isUnlabeled() {
        return (this.numLabeledPoints == 0);
    }

    /**
     * ONly the number of labeled points provided to this cluster for training
     * Note that this number should equal the sum of this.labelFrequencies
     * @return number of labeled data points provided to this cluster
     */
    public final int getNumLabeledPoints() {
        return this.numLabeledPoints;
    }

    /**
     * Total number of points provided to this cluster
     * @return number of labeled and unlabeled points contributing to this cluster
     */
    public final int size() {
        return this.numTotalPoints;
    }

    /**
     * Return the label frequencies gathered from training data to this cluster
     * @return vote array
     */
    public final double[] getVotes() {
        return this.labelFrequencies;
    }

    /**
     * Return the predicted label for the cluster. This is provided by the embedded classifier (if exists and trained),
     * otherwise by the label frequencies gathered from training data to this cluster
     * @param x Instance to pass to embedded discriminant classifier
     * @return vote array
     */
    public final double[] getVotesForInstance(Instance x) {
        if ((this.embeddedClassifier == null) || (this.embeddedClassifier.trainingWeightSeenByModel() < 2)) {
            return getVotes();
        }
        double votes[] = this.embeddedClassifier.getVotesForInstance(x);
        if (this.excludeOutlierVoting) { // Majority class will always vote 'outlier'
            int outlierIdx = this.instances.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
            if (outlierIdx > 0 && outlierIdx < votes.length) {
                votes[outlierIdx] = 0;
            }
        }
        return votes;
    }
    
    /**
     * Get the probability of a given class based on observed frequencies
     *
     * @param labelIndex class index
     * @return probability that class labelIndex is the true class of this cluster group
     */
    public final double getClassProbability(int labelIndex) {
        double ret = 0;
        double sum = weka.core.Utils.sum(labelFrequencies);
        if ((sum > 0) && (labelIndex < this.labelFrequencies.length)) {
            ret = this.labelFrequencies[labelIndex] / sum;
        }
        return ret;
    }

    /**
     * Get number of classes known by this cluster
     *
     * @return
     */
    public final int getNumClasses() {
        int ret = 0;
        for (int i = 0; i < this.labelFrequencies.length; ++i) {
            if (this.labelFrequencies[i] > 0) {
                ret++;
            }
        }
        return ret;
    }

    /**
     * 
     * @param x data instance in question for comparison to this distribution.
     * @return probability that point x is a member of this cluster
     */
    @Override
    public final double getInclusionProbability(Instance x) {
        int labelIndex = x.classIndex();
        double[] values = x.toDoubleArray();
        return getInclusionProbability(values, labelIndex);
    }
    
        /**
     * 
     * @param x data instance in question for comparison to this distribution.
     * @param ignore index to ignore (e.g. class index)
     * @return probability that point x is a member of this cluster
     */
    public final double getInclusionProbability(double[] x, int ignore) {
        double ret = 1;
        int ignoreIdx = (ignore >= 0) ? ignore : this.instances.classIndex();
        for(int i = 0; i < x.length; ++i) {
            if (i == ignoreIdx) { continue; }
            double pxi = Math.max(getComponentProbability(x[i], i), LAPLACIAN_SMOOTHING_CONSTANT);
            ret *= pxi;
        }
        return ret;
    }
    
    public final double getComponentProbability(double x, int idx) {
        double ret = LAPLACIAN_SMOOTHING_CONSTANT;
        if (this.symbolFrequencies[idx] == null) {
            //double mean = this.getCenter()[idx];
            double stddev = Math.sqrt(this.variances[idx]);
            if (stddev <= LAPLACIAN_SMOOTHING_CONSTANT) {
                return (Math.abs(x - centroid[idx]) < LAPLACIAN_SMOOTHING_CONSTANT) ? 0.95 : LAPLACIAN_SMOOTHING_CONSTANT;
            }
            double Z = Math.abs(x - centroid[idx]) / stddev;
            //double bucketRadius = 1.0 / Math.sqrt(Math.max(this.numTotalPoints, 5)); // n^(-1/2)
            //double lowerBound = weka.core.FastStats.normalProbability(Z - bucketRadius);
            //double upperBound = weka.core.FastStats.normalProbability(Z + bucketRadius);
            //ret = Math.abs(upperBound - lowerBound); // should not be necessary, but just in case
            ret = Math.max(normalPDF(Z, 0.0, 1.0), LAPLACIAN_SMOOTHING_CONSTANT);
            //ret = Math.max(this.normalERF(x, this.getCenter()[idx], this.variances[idx]), LAPLACIAN_SMOOTHING_CONSTANT);
        } else if ((this.numTotalPoints > 0) && (x > 0) && (symbolFrequencies[idx].length > (int) x)) {
            double denominator = 0;
            for (Double d : this.symbolFrequencies[idx]) {
                denominator += d;
            }
            assert (denominator > 0) : "Denominator is zero!";
            double numerator = Math.max(0, this.symbolFrequencies[idx][(int) x]);
            ret = numerator / denominator;
        } else if (Math.abs(x - centroid[idx]) < weka.core.Utils.SMALL) {
            ret = 0.99;
        }
            
        return ret;
    }
    
    /**
     * Gaussian Probability Density Function (PDF)
     * 
     * @param x
     * @param mean
     * @param variance (Not standard deviation!)
     * @return 
     */
    public final double normalPDF(double x, double mean, double variance) {
        double ret = (x == mean) ? 0.95 : 0;
        if (variance > 0) {
            double Z = (x - mean) / Math.sqrt(variance);
            ret = Math.exp(-0.5 * Z * Z) / GAUSSIAN_DIVISOR;
        }
        return ret;
    }
    
        /**
     * Gaussian Probability Density Function (PDF)
     * 
     * @param x
     * @param mean
     * @param variance (Not standard deviation!)
     * @return 
     */
    public final double normalERF(double x, double mean, double variance) {
        double ret = (x == mean) ? 0.95 : 0;
        if (variance > 0) {
            double Z = (x - mean) / Math.sqrt(variance);
            ret = weka.core.FastStats.errorFunctionComplemented(Z);
        }
        return ret;
    }
    
    /**
     * Discrete Probability Mass Function (CMF)
     * @param x
     * @param sym symbol frequency count
     * @return 
     */
    public final double discretePMF(double x, double sym[]) {
        double ret = LAPLACIAN_SMOOTHING_CONSTANT;
        if ((this.numTotalPoints > 0) && (symbolFrequencies != null) && (x > 0) && (symbolFrequencies.length > (int) x)) {
            double numerator = Math.max(0,sym[(int) x]);
            ret = numerator / this.numTotalPoints; 
        }
        return ret;
    }
    
    
    /**
     * Use the multi-variate Gaussian probability estimation approach to determine probability x is a member of this cluster.
     * This is not the same as a true Mahalanobis distance derived from a full covariance matrix as we take the heuristic shortcut
     * of assuming variable independence (rho = 0), and thus have an axis-parallel hyper-ellipse (i.e. diagonal covariance matrix)
     * under the same 'naive' assumption as in Naive Bayes. The computational speed-up here is worth the reduced accuracy.
     * @param x instance in question
     * @return probability that point x is a member of this cluster
     * @deprecated 
     */
    @Deprecated
    public double getInclusionMVGProbability(Instance x) {
        
        double[] diagonalCovariance = this.getVariances();
        int degreesOfFreedom = 0;
        double axisParallelSquaredMahalanobusDistance = 0;
        // Compute Mahalanobis (axis-parallel estimation) distance and tally degrees of freedom
        for(int i = 0; i < centroid.length; ++i) {
            if (i == x.classIndex()) { continue; }
            double attributeDelta = (this.symbolFrequencies[i] == null) ? x.value(i) - centroid[i] : (Math.abs(x.value(i) - centroid[i]) < 1e-32) ? 0 : 1;
            if ((diagonalCovariance[i] != 0) && Double.isFinite(diagonalCovariance[i])) {
                axisParallelSquaredMahalanobusDistance += attributeDelta * attributeDelta / diagonalCovariance[i];
            }
            degreesOfFreedom++;
        }
        
        // Zero distance means exact match, so stop working on more computations...
        if ((axisParallelSquaredMahalanobusDistance == 0) && (degreesOfFreedom > 0)) { return 1.0; } 
        
        // Compute the determinant of the Covariance matrix (which is O(N) if we only have the diagonal of the matrix)
        double covarianceDet = 1;
        for(int i = 0; i < diagonalCovariance.length; ++i) {
            if (i == x.classIndex()) { continue; }
            if ((diagonalCovariance[i] != 0) && Double.isFinite(diagonalCovariance[i])) {
                covarianceDet *= diagonalCovariance[i];
            }
            else {
                covarianceDet *= weka.core.Utils.SMALL;
            }
        }
        if (covarianceDet == 0) { covarianceDet = weka.core.Utils.SMALL / 1000.0; } // Safety
        double score = Math.exp(0.0 - 0.5 * axisParallelSquaredMahalanobusDistance) / (Math.pow(2.0 * Math.PI, degreesOfFreedom / 2.0) * Math.sqrt(Math.abs(covarianceDet)));
        return (1.0 - weka.core.FastStats.chiSquaredProbability(score, degreesOfFreedom));
    }
    

//  
//    /**
//     * Use outlier-criteria (selectable strategy) for determining if a data point is an outlier (unlikely member) of cluster
//     *
//     * @param x data point in question
//     * @return true if x is an outlier w.r.t. this cluster
//     */
//    public boolean isOutlier(Instance x) {
//        boolean ret = false;
//        double p = this.getInclusionProbability(x);
//        ret = this.isOutlier(p);
//        return ret;
//    }

    /**
     * Computes the Chauvenet Criterion limit for this cluster
     * @return 
     */
    public final double getChauvenetLimit() {
        double N = Math.min(this.numTotalPoints, 1.0 / (this.alphaAdjustmentWeightOption.getValue() + 1e-9));
        double degreesOfFreedom = Math.max(2, this.instances.numAttributes() - 1);
        //double bucketRadius = 3.0 / Math.sqrt(Math.max(N, 5)); // n^(-1/2)
        //double p = 1 / (4 * N);
        //double chauvenetZ = weka.core.FastStats.normalInverse(p);
        //double lowerBound = weka.core.FastStats.normalProbability(chauvenetZ - bucketRadius);
        //double upperBound = weka.core.FastStats.normalProbability(chauvenetZ + bucketRadius);
        //double binnedResult = Math.pow(upperBound - lowerBound, degreesOfFreedom);
        //double ChiSqrdResult = weka.core.FastStats.chiSquaredProbability( Math.abs(weka.core.FastStats.normalInverse(p) * degreesOfFreedom), degreesOfFreedom - 1);
        //double ChiSqrdResult = weka.core.FastStats.chiSquaredProbability( Math.abs(weka.core.FastStats.normalInverse(p * degreesOfFreedom) ), degreesOfFreedom - 1);
        double powerResult =  Math.pow(1.0 / (2.0 * Math.max(N,32)), degreesOfFreedom);
        return powerResult;
    }

    /**
     * Use outlier-criteria (selectable strategy) for determining if a data point is an outlier (unlikely member) of cluster
     *
     * @param x Instance for comparison to see if it is an outlier to this cluster
     * @return true if x is an outlier w.r.t. this cluster
     */
    public final boolean isOutlier(Instance x) {
        boolean ret;
        double p = this.getInclusionProbability(x);
        switch (this.outlierDefinitionStrategyOption.getChosenIndex()) {
            case 0: //Use Chauvenet's Criteria to determine outlier standing of the data point for this cluster.
                ret = (p < getChauvenetLimit());
                break;
            case 1: // use Perceptron
                double[] v = embeddedClassifier.getVotesForInstance(x);
                try {
                    weka.core.Utils.normalize(v);
                } catch(Exception e) {}
                int oIdx = x.dataset().classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
                double po = (v.length > oIdx) ? v[oIdx] : 0;
                if (po <= 0) { v[oIdx] = 0; }
                int h = weka.core.Utils.maxIndex(v);
                double ph = v[h];
                double margin =  (po - ph);
                ret = (po > ph) && (margin > (2.0 / v.length));
                break;
            case 2: // 2.5 sigma
                ret = (p < weka.core.FastStats.normalProbability(2.5));
                break;
            case 3: // 3 sigma
                ret = (p < weka.core.FastStats.normalProbability(3));
                break;
            case 4: // 6 sigma 
                ret = (p < weka.core.FastStats.normalProbability(6));
                break;
            case 5: // cheat
                ret = p > 0.5;
                break;
            default:
                ret = p < weka.core.FastStats.normalProbability(2.5);
        }
        return ret;
    }
    
    
    /**
     * Used to provide additional labeling information via SSL methods
     * @param label class label to cast vote for
     * @param w weight to add
     */
    public void addLabeling(int label, double w) {
        if ((label > 0) && (label < this.labelFrequencies.length)) {
            this.labelFrequencies[label] += w;
        }
    }
    
    /**
     * * Used to provide additional labeling information via SSL methods
     * @param label class label to cast vote for
     * @param w weight to remove
     */
    public void removeLabeling(int label, double w) {
        if ((label > 0) && (label < this.labelFrequencies.length)) {
            this.labelFrequencies[label] -= w;
        }
    }
    

    
    /**
     * Add a data point instance to this cluster
     *
     * @param x
     */
    final public void addInstance(Instance x) {
        safeInit(x);
        this.numTotalPoints++;
        this.numLabeledPoints += (x.weight() > 0.9999) ? 1 : 0;
        this.labelFrequencies[(int) x.classValue()] += x.weight(); //non-training data has a weight of zero
        this.gtLabelFrequencies[(int) x.classValue()]++; // For non-decision metrics only
        //Select strategy for on-line *-means (Any means)
        switch (updateStrategyOption.getChosenIndex()) {
            case 0:
                this.addInstanceGrimson(x);
                break;
            case 1:
                this.addInstanceViaShephard(x);
                break;
            case 2:
                this.instances.add(x);
                return;
            default:
                System.err.println("Invalid addInstance strategy");
        }
        recompute();
    }

    /**
     * Inverse process of adding instance
     *
     * @param x
     */
    final public void removeInstance(Instance x) {
        safeInit(x);
        this.numLabeledPoints -= (int) Math.ceil(x.weight());
        this.labelFrequencies[(int) x.classValue()] -= x.weight(); //non-training data has a weight of zero
        this.gtLabelFrequencies[(int) x.classValue()]--; // For non-decision metrics only
        this.numTotalPoints--;
        
        //Select strategy for on-line *-means
        switch (updateStrategyOption.getChosenIndex()) {
            case 0:
                this.removeInstanceGrimson(x);
                break;
            case 1:
                this.removeInstanceViaShephard(x);
                break;
            case 2:
                this.instances.remove(x);
                return;
            default:
                System.err.println("Invalid removeInstance strategy");
        }
        recompute();
    }

    /**
     * Add instance to this cluster
     * Note that caller function takes care of object recompute() call. We just need to update the specific changes here
     * @param x
     */
    final protected void addInstanceGrimson(Instance x) {
        double newPoint[] = x.toDoubleArray();
        double newVariances[] = new double[centroid.length];
        double alpha = this.alphaAdjustmentWeightOption.getValue();
        double deviation;
        double p;
        for (int i = 0; i < centroid.length; ++i) {
            p = getComponentProbability(newPoint[i], i);
            if (this.symbolFrequencies[i] == null) { // update numeric attribute tracking
               rho[i] = p * alpha;
               centroid[i] = (1.0 - rho[i]) * centroid[i] + rho[i] * newPoint[i];
               deviation = newPoint[i] - centroid[i];
               newVariances[i] = (1.0 - rho[i]) * this.variances[i] + rho[i] * deviation * deviation;
            } else { // update nominal attribute tracking
                rho[i] = (this.entropies[i] <= 0) ? alpha : p * alpha;
                for(int j = 0; j < this.symbolFrequencies[i].length; ++j) {
                    this.symbolFrequencies[i][j] *= (1.0 - rho[i]);
                }
                this.symbolFrequencies[i][(int) newPoint[i]] += rho[i];
                centroid[i] = weka.core.Utils.maxIndex(this.symbolFrequencies[i]);
                newVariances[i] = 1;
            }
        }
        this.setCenter(centroid);
        this.variances = newVariances;
    }

    /**
     * Inverse process of adding instance
     * Note that caller function takes care of object recompute() call. We just need to update the specific changes here
     * @param x
     */
    final protected void removeInstanceGrimson(Instance x) {
        double newPoint[] = x.toDoubleArray();
        double alpha = this.alphaAdjustmentWeightOption.getValue();
        double deviation;
        double p;
        for (int i = 0; i < centroid.length; ++i) {
            p = getComponentProbability(newPoint[i], i);
            if (this.symbolFrequencies[i] == null) { // update numeric attribute tracking
               rho[i] = p * alpha;
               deviation = newPoint[i] - centroid[i];
               if (rho[i]  != 1 && newPoint[i] != 0) { centroid[i] = centroid[i] / (1.0 - rho[i] ) - rho[i]  / newPoint[i]; }// inverse operation from adding
               if (deviation > 0) { this.variances[i] = this.variances[i] / (1.0 - rho[i] ) -  rho[i]  / (deviation * deviation);} // inverse operation from adding
            } else { // update nominal attribute tracking
                rho[i]  = (this.entropies[i] <= 0) ? alpha : p * alpha;
                double symFreq[] = this.symbolFrequencies[i];
                for(int j = 0; j < symFreq.length; ++j) {
                    this.symbolFrequencies[i][j] /= (1.0 - rho[i] );
                }
                this.symbolFrequencies[i][(int) newPoint[i]] -= rho[i] ;
                centroid[i] = weka.core.Utils.maxIndex(symFreq);
            }
        }
        this.setCenter(centroid);
    }

    /**
     * Add instance to this cluster Computes cluster statistics using an adaptation of Shepherd's scalar on-line method
     * Note that caller function takes care of object recompute() call. We just need to update the specific changes here
     * @param x
     */
    final protected void addInstanceViaShephard(Instance x) {
        // multi-dimensional extension to Data Analysis 4th Ed Ch. 2 (Shepherd)
        assert (numTotalPoints > 1) : " Too few points to compute metrics";
        double newPoint[] = x.toDoubleArray();
        for (int i = 0; i < centroid.length; ++i) {
            if (this.symbolFrequencies[i] == null) {
                double d = newPoint[i] - centroid[i];
                centroid[i] = centroid[i] + d / (this.numTotalPoints); // 1 / (n + 1) implicit since we already did n++ in parent function
                rho[i] = rho[i] + d * (newPoint[i] - centroid[i]);
                this.variances[i] = rho[i] / (this.numTotalPoints - 1);
            } else {
                int newVal = (int) newPoint[i];
                if (newVal < this.symbolFrequencies[i].length) {
                    this.symbolFrequencies[i][(int) newPoint[i]]++;
                }
                centroid[i] = weka.core.Utils.maxIndex(symbolFrequencies[i]);
            }
        }
        this.setCenter(centroid);
    }

    /**
     * Inverse process of adding instance Note that caller function takes care of object recompute() call. We just need to
     * update the specific changes here
     *
     * @param x
     */
    final protected void removeInstanceViaShephard(Instance x) {
        // multi-dimensional extension to Data Analysis 4th Ed Ch. 2 (Shepherd)
        if (this.numTotalPoints > 0) {
            double runningDeviation = this.getCenterDistance(x);
            double newPoint[] = x.toDoubleArray();
            for (int i = 0; i < centroid.length; ++i) {
                if (this.symbolFrequencies[i] == null) {
                    double attributeDist = newPoint[i] - centroid[i];
                    centroid[i] = centroid[i] - attributeDist / this.numTotalPoints;
                } else {
                    int newVal = (int) newPoint[i];
                    if (newVal < this.symbolFrequencies[i].length) {
                        this.symbolFrequencies[i][(int) newPoint[i]]--;
                    }
                    centroid[i] = weka.core.Utils.maxIndex(symbolFrequencies[i]);
                }
            }
            this.setCenter(centroid);
            this.runningSumOfSquares -= runningDeviation * this.getCenterDistance(x);
        }
    }

    /**
     *
     * @param x
     * @return
     */
    @Override
    final public double getCenterDistance(Instance x) {
        if (this.distanceStrategyOption.getChosenIndex() == 13) {
         return 1.0 - this.getInclusionProbability(x);   
        } else {
            double[] src = x.toDoubleArray();
            return VectorDistances.distance(src, centroid, this.instances, this.distanceStrategyOption.getChosenIndex());
        }
    }

    /**
     *
     * @param x
     * @return
     */
    final public double getCenterDistance(double[] x) {
        if (this.distanceStrategyOption.getChosenIndex() == 13) {
         return 1.0 - this.getInclusionProbability(x,-1);   
        } else {
            return VectorDistances.distance(x, centroid, this.instances, this.distanceStrategyOption.getChosenIndex());
        }
    }
    
    /**
     *
     * @param other cluster
     * @return
     */
    public double getCenterDistance(Riffle other) {
        if (this.distanceStrategyOption.getChosenIndex() == 13) {
            DenseInstance x = new DenseInstance(0,other.centroid);
            x.setDataset(this.instances);
            return 1.0 - this.getInclusionProbability(x);   
        } else {
            return VectorDistances.distance(other.centroid, centroid, this.instances, this.distanceStrategyOption.getChosenIndex());
        }
    }
    
    /**
     * Set pre-computed information fields
     */
    public final void recompute() {
        //double[] axisParallelCovariance = this.getVariances();
        double entropy;
        for(int i = 0; i < this.entropies.length; ++i) {
            if (this.symbolFrequencies[i] == null) {
                if (this.variances[i] == 0) {
                    entropy = 0;
                } else {
                    entropy = 0.5 * Math.log(PI2E * this.variances[i]) / Math.log(2); // Gaussian entropy function
                }
                this.entropies[i] = entropy;
            } else {
                double[] symbolFreq = this.symbolFrequencies[i];
                entropy = 0.0;
                double normalizer = 0;
                for (double v : symbolFreq) {
                    normalizer += v;
                }
                for (double v : symbolFreq) {
                    double p = v / normalizer;
                    entropy -= (p > 0) ? p * Math.log(p) / Math.log(2) : 0;
                }
                this.entropies[i] = entropy;
            }
        }
        double newRadius = 0;
//        for(int i = 0; i < axisParallelCovariance.length; ++i) {
//            if (this.instances.classIndex() != i) {
//                newRadius += Math.max(axisParallelCovariance[i], weka.core.Utils.SMALL);
//            }
//        }
//        this.setRadius(Math.sqrt(newRadius));
        // Per Dr. Masud's approach, radius is the distance of the furthest point
        if (instances != null && !instances.isEmpty()) {
            for(Instance x : instances) {
                double localDist = getCenterDistance(x);
                if (Double.isFinite(localDist)) {
                    newRadius = Math.max(newRadius, localDist);
                }
            }
        }
        setRadius(newRadius);
        this.setGroundTruth(weka.core.Utils.maxIndex(this.labelFrequencies));
    }
    
    /**
     * Set pre-computed information fields
     * @return 
     */
    public final double recomputeAll() {
        if (this.instances != null) {
            Arrays.fill(this.gtLabelFrequencies, 0);
            Arrays.fill(this.labelFrequencies, 0);
            this.numTotalPoints = instances.size();
            this.numLabeledPoints = 0;
            if (!this.instances.isEmpty()) {
                // double[] clusterCentroid = this.getCenter();
                double[] clusterVariance = this.getVariances();
                for (int i = 0; i < centroid.length; ++i) {
                    centroid[i] /= (double) this.instances.size() + 1.0;
                }
                for (double[] sf : this.symbolFrequencies) {
                    if (sf != null) {
                        Arrays.fill(sf, 0);
                    }
                }
                for (Instance x : this.instances) { // Pre-populate univeral cluster with data points
                    if (x == null) {
                        System.out.println("Sieve::MaximizationStep() - x is NULL!");
                        continue;
                    }
                    this.gtLabelFrequencies[(int) x.classValue()]++;
                    this.labelFrequencies[(int) x.classValue()] += x.weight();
                    this.numLabeledPoints += x.weight();
                    double[] xValues = x.toDoubleArray();
                    for (int i = 0; i < xValues.length; ++i) {
                        double val = xValues[i];
                        centroid[i] += val / ((double) this.instances.size() + 1.0);
                        if ((this.symbolFrequencies[i] != null) && (val < this.symbolFrequencies[i].length)) {
                            this.symbolFrequencies[i][(int) val]++;
                        }
                    }
                }// for

                // Set 'centroid' to 'mode' (most frequent symbol) for nominal data:
                for (int i = 0; i < this.symbolFrequencies.length; ++i) {
                    if (this.symbolFrequencies[i] != null) {
                        centroid[i] = weka.core.Utils.maxIndex(this.symbolFrequencies[i]);
                    }
                }
                setCenter(centroid); // temporary - start with standard gaussian, gets updated below
                // The cluster class uses an incremental heuristic, but we want to start out as pure as possible, so
                // we use the 2-Pass method for computing sample variance (per dimension)
                double n = instances.size();
                if (n > 1) {
                    double[] cep = new double[centroid.length];
                    Arrays.fill(cep, 0);
                    Arrays.fill(clusterVariance, 0);
                    for (Instance x : this.instances) {
                        if (x == null) {
                            System.out.println("Riffle::recompute() - x is null!");
                            continue;
                        }
                        double[] xValues = x.toDoubleArray();
                        for (int i = 0; i < xValues.length; ++i) {
                            double delta = (this.symbolFrequencies[i] == null) ? centroid[i] - xValues[i] : (Math.abs(centroid[i] - xValues[i]) < 1e-32) ? 1 : 1e-20;
                            cep[i] += delta;
                            clusterVariance[i] += delta * delta; // Statistical Variance
                        }
                    }
                    for (int i = 0; i < clusterVariance.length; ++i) {
                        clusterVariance[i] = (clusterVariance[i] - cep[i] * cep[i] / n) / (n - 1);
                    }
                    setVariances(clusterVariance);
                } // end if (enough data for variance)
            } // end if(!instances.empty)
            recompute();
        } // end if(!instances null)
        return getRadius() * getEntropy();
    }
      

    /**
     *
     * @return purity (0-1) of cluster
     */
    public double getTruePurity() {
        double sum = 0.0;
        double max = 0.0;
        for (int i = 0; i < this.gtLabelFrequencies.length; ++i) {
            sum += this.gtLabelFrequencies[i];
            max = Math.max(max, this.gtLabelFrequencies[i]);
        }
        return (sum > 0) ? max / sum : 1.0;
    }

    /**
     *
     * @return purity (0-1) of cluster
     */
    public double getTrueEntropy() {
        if (this.numTotalPoints < 0) {
            return 0;
        }
        double entropy = 0.0;
        for (int i = 0; i < this.gtLabelFrequencies.length; ++i) {
            double p = this.gtLabelFrequencies[i] / this.numTotalPoints;
            entropy -= (p > 0) ? p * Math.log(p) / LOG2 : 0;
        }
        return entropy;
    }

           
    /**
     *
     * @return purity (0-1) of cluster
     */
    public double getPurity() {
        double sum = 0.0;
        double max = 0.0;
        for (int i = 0; i < this.labelFrequencies.length; ++i) {
            sum += this.labelFrequencies[i];
            max = Math.max(max, this.labelFrequencies[i]);
        }
        return (sum > 0) ? max / sum : 1.0;
    }

    /**
     *
     * @return purity (0-1) of cluster
     */
    public final double getEntropy() {
        if (this.numLabeledPoints < 0) {
            return 0;
        }
        double entropy = 0.0;
        for (int i = 0; i < this.labelFrequencies.length; ++i) {
            double p = this.labelFrequencies[i] / this.numLabeledPoints;
            entropy -= (p > 0) ? p * Math.log(p) / LOG2 : 0;
        }
        return entropy;
    }

    /**
     * Absorb another cluster
     * package accessible
     * @param other the cluster to absorb
     */
    void merge(Riffle other) {
        double w0 = this.getWeight() +  other.getWeight();
        if (Double.isNaN(w0) || w0 == 0) { w0 = 1; }
        double w1 = this.getWeight() / w0;
        double w2 = other.getWeight() / w0;
        
        for(int i = 0; i < other.labelFrequencies.length && i < this.labelFrequencies.length; ++i) {
            this.labelFrequencies[i] += other.labelFrequencies[i];
            this.gtLabelFrequencies[i] += other.gtLabelFrequencies[i];
        }
        for(int i = 0; i < this.variances.length && i < other.variances.length; ++i) {
            this.variances[i] += other.variances[i];
        }
        for(int i = 0; i < this.symbolFrequencies.length && i < other.symbolFrequencies.length; ++i) {
            if (this.symbolFrequencies[i] != null && other.symbolFrequencies[i] != null) {
                for(int j = 0; j < this.symbolFrequencies[i].length && j < other.symbolFrequencies[i].length; ++j) {
                    this.symbolFrequencies[i][j] += other.symbolFrequencies[i][j];
                }
            }
        }
        double[] attribWeights = new double[this.symbolFrequencies.length];
        double[] newCenter = new double[centroid.length];
        for(int i = 0; i < attribWeights.length; ++i) {
            attribWeights[i] = w1 * this.instances.attribute(i).weight() + w2 * other.instances.attribute(i).weight();
            newCenter[i] = w1 * centroid[i] + w2 * other.centroid[i];
        }
        weka.core.Utils.normalize(attribWeights);
        this.setAttributeWeights(attribWeights);
        this.setCenter(newCenter);
        
        // Todo - combine rho values        
        
        this.numLabeledPoints += other.numLabeledPoints;
        this.numTotalPoints += other.numTotalPoints;
        this.runningSumOfSquares += other.runningSumOfSquares;
        this.setGroundTruth(weka.core.Utils.maxIndex(this.gtLabelFrequencies));
        //this.setRadius(w1 * this.getRadius() + w2 * other.getRadius());
        this.setWeight(w0 / 2.0);
        recompute();
    }
    
    /**
     * Converts cluster to an instance
     * @return Instance version of the centroid
     */
    public Instance toInstance() {
        DenseInstance ret = new DenseInstance(0.0, centroid);
        ret.setWeight(0.0);
        ret.setDataset(this.instances);
        return ret;
    }
    
    public void penalize() {
        double newWeight = this.getWeight() * (1.0 - this.alphaAdjustmentWeightOption.getValue());
        this.setWeight(newWeight); // penalize
    }
    
    public void reward() {
        double newWeight = this.getWeight() + this.alphaAdjustmentWeightOption.getValue();
        this.setWeight(newWeight); // reward
    }
    
    /**
     * Sanity check and initialization of dynamic fields
     *
     * @param x
     */
    protected final void safeInit(Instance x) {
        if (this.embeddedLearnerOption.getValueAsCLIString().contains("Majority class")) {
            this.excludeOutlierVoting = true;
        }
        if (centroid == null) {
            centroid = x.toDoubleArray();
        }
        if (this.instances == null) {
            prepareEmbeddedClassifier();
            ArrayList<Attribute> attribs = new ArrayList<>();
            this.symbolFrequencies = new double[x.dataset().numAttributes()][];
            for (int i = 0; i < x.dataset().numAttributes(); ++i) {
                Attribute a = (Attribute) x.dataset().attribute(i).copy();
                if (i == x.classIndex()) { a.setWeight(0.0);} else { a.setWeight(1.0);}
                switch(a.type()) {
                    case Attribute.STRING:
                    case Attribute.NOMINAL:
                        //UnsafeUtils.setAttributeRange(a, x.value(i), x.value(i));
                        this.symbolFrequencies[i] = new double[a.numValues()];
                        break;
                    case Attribute.NUMERIC:
                    case Attribute.RELATIONAL:
                    case Attribute.DATE:
                    default:
                       // UnsafeUtils.setAttributeRange(a, x.value(i), x.value(i));
                        this.symbolFrequencies[i] = null;
                }
                attribs.add(a);
            }
            this.instances = new Instances("ClusterData", attribs, 1);
            this.instances.setClassIndex(x.classIndex());
            
        } 
//        else {
//            for (int i = 0; i < x.dataset().numAttributes() && i < this.header.numAttributes(); ++i) {
//                double val = x.value(i);
//                Attribute a = this.header.attribute(i);
//                // expand range as necessary
//                if (val < a.getLowerNumericBound() || val > a.getUpperNumericBound()){
//                    UnsafeUtils.setAttributeRange(a, Math.min(val,a.getLowerNumericBound()), Math.max(val,a.getUpperNumericBound()));
//                }
//                // increase frequency counts if new string value is encountered
//                if (a.type() == Attribute.STRING && (val >= Math.max(this.symbolFrequencies[i].length, a.numValues()))) {
//                    double newArray[] = new double[Math.max(this.symbolFrequencies[i].length, a.numValues())];
//                    Arrays.fill(newArray, 0);
//                    for(int j = 0; j <= this.symbolFrequencies[i].length; j++) {
//                        newArray[j] = this.symbolFrequencies[i][j];
//                    }
//                    this.symbolFrequencies[i] = newArray;
//                }
//            }
//        }
        if (this.variances == null) {
            this.variances = new double[x.numAttributes()];
            Arrays.fill(this.variances, 1);
        }
        if (this.entropies == null) {
            this.entropies = new double[x.numAttributes()];
            Arrays.fill(this.entropies, 0);
        }
        if (this.labelFrequencies == null) {
            this.labelFrequencies = new double[x.numClasses()];
            Arrays.fill(this.labelFrequencies, 0);
        }
        if (this.gtLabelFrequencies == null) {
            this.gtLabelFrequencies = new double[x.numClasses()];
            Arrays.fill(this.gtLabelFrequencies, 0);
        }
        if (this.rho == null) {
            this.rho = new double[x.numAttributes()];
            Arrays.fill(this.rho, 0);
        }
    }

    /**
     * The SizeOfAgent method returns a value or -1 many times, so this override assures at least some estimate
     * using intrinsic knowledge of the object structure.
     * @return Estimated numbed of bytes used by this object for data
     */
    @Override
    public int measureByteSize() {
        int ret = super.measureByteSize();
        if (ret <= 0) {
            ret = 0;
            int numAttributes = (this.variances == null)   ? 0 : this.variances.length;
            int numClasses = (this.gtLabelFrequencies == null) ? 0 : this.gtLabelFrequencies.length;
            for(double[] a : this.symbolFrequencies) {
                ret += (a == null) ? 0 : a.length * Double.SIZE / 8;
            }
            ret += 5 * Double.SIZE  / 8;
            ret += 2 * Integer.SIZE / 8;
            ret += (numAttributes * 3 + numClasses * 2) * Double.SIZE / 8;
        }
        return ret;
    }
    
    
    @Override
    public int compareTo(SphereCluster other) {
        int ret = 0;
        if (this.getId() < other.getId()) {
            ret = 1;
        } else if (this.getId() > other.getId()) {
            ret = -1;
        }
        return ret;
    }
    
    /**
     *  Custom fast hashing code
     * @return unique ID of this cluster for hashing
     */
    @Override
    public int hashCode() {
        return (int) (this.getId() % 100_000);
    }
    
    @Override
    public boolean equals(Object other) {
        if (other == null) { return false;}
        return this.getId() == ((SphereCluster) other).getId();
    }
}
