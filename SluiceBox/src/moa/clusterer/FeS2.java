/*
 *    FeS2.java
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
package moa.clusterer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import static java.util.Arrays.parallelSort;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import moa.classifiers.functions.Perceptron;
import moa.cluster.Clustering;
import moa.cluster.Riffle;
import moa.core.tuples.NearestClusterTuple;
import moa.core.tuples.ValIdxTupleType;
import moa.clusterers.AbstractClusterer;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.core.VectorDistances;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Finding Entities in Sub-Space (FeS2)
 * Finding Entities in Salient Streams
 * Following Evolving Stream Sets
 * Following Evolving Salient Sets
 * Re-InForcement Learning (rifling)
 * This class was originally designed for use as part of Brandon Parker's
 * Dissertation work.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 3 $
 */

final public class FeS2 extends AbstractClusterer {
    private static final long serialVersionUID = 1L;
    protected static final DateFormat iso8601FormatString = new SimpleDateFormat("yyyyMMdd'T'HHmmssSSS");
    
    public final IntOption minimumClusterSizeOption = new IntOption("minimumClusterSize", 'm',
            "Minimum size for a cluster to be used as a labeled cluster",
            5, 1, Integer.MAX_VALUE);
    
    public final IntOption minimumNumberOfClusterSizeOption = new IntOption("minNumClusters", 'k',
            "Minimum number of clusters to keep",
            10, 1, Integer.MAX_VALUE);
    
    public final IntOption maximumNumberOfClusterSizeOption = new IntOption("maxNumClusters", 'K',
            "Minimum number of clusters to keep",
            150, 2, Integer.MAX_VALUE);
    
    public final IntOption clustersPerLabelOption = new IntOption("ClustersPerLabel", 'E',
            "Preferred number of clusters to keep per known label, subject to min/max num cluster constraints",
            5, 1, Integer.MAX_VALUE);

    public final FloatOption learningRateAlphaOption = new FloatOption("learningRateAlpha", 'a',
            "Learning Rate. Sets how much a new success/failure affects the weighting. Small means very little effect and a larger 'memory' for the weights",
            0.01, 0.0, 0.5);
    
    public final FloatOption initialClusterWeightOption = new FloatOption("initialClusterWeight", 'w',
            "Weight to set for cluster upon intial creation (first data point)",
            1.0, 0.0, 1.0);
    
//    public final FloatOption pruneThresholdOption = new FloatOption("pruneThreshold", 'e',
//            "Minimum weight a cluster can have before it is pruned",
//            0.001, 0.0, 1.0);

    public final MultiChoiceOption updateStrategyOption = new MultiChoiceOption("UpdateStrategy", 'u',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Stauffer-Grimson", "Shephard"},
            new String[]{"Gaussian update based on Stauffer and Grimson (1999)", "Robust update to momemts based on Shephard (1996)"},
            0);
  
    public final MultiChoiceOption outlierDefinitionStrategyOption = new MultiChoiceOption("outlierDefinitionStrategy", 'o',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Chauvenet", "Learned", "2.5-sigma", "3-sigma", "6-sigma","oracle"},
            new String[]{"Chauvenet", "Used trained Perceptron", "2.5-sigma", "3-sigma", "6-sigma","Cheat and use ground truth (for unit testing purposes)"},
            1);

//    public final MultiChoiceOption distanceNormStrategyOption = new MultiChoiceOption("distanceNormStrategy", 'n',
//            "Set strategy for attribute normalization for distances.",
//            new String[]{"none", "weight", "variance", "weight/variance", "random"},
//            new String[]{"none", "weight", "variance", "weight/variance", "random"},
//            0);
    
    public final MultiChoiceOption subspaceStrategyOption = new MultiChoiceOption("subspaceStrategy", 's',
            "Set strategy subspace selection.",
            new String[]{"none", "K-L Curve", "K-L Norm", "Variance", "Info Gain Rank", "random"},
            new String[]{"none", "Keep only attributes right of the Curve of ordered K-L divergances", "Normalize weigh by K-L Ranking", "Normalized Variance ranking", "Normalized rank based on attribute information gain", "Choose Random Weights"},
            0);
    
    public final MultiChoiceOption votingStrategyOption = new MultiChoiceOption("votingStrategy", 'V',
            "Set strategy for tabulating voting for predicted label among the online clusters (i.e. 'neighbor' exemplars).",
            new String[]{"1-NN", "G-NN", "GW-NN","GWP-NN","GN-NN","GNP-NN", "GNPW-NN", "GWS-NN"},
            new String[]{"1-NN", // 0
                "Global  k-NN", // 1
                "Globally probability-weighted k-NN", // 2    
                "Globally probability-utility-weighted k-NN", // 3
                "Globally normalized k-NN", // 4
                "Globally normalized probability-weighted k-NN", // 5
                "Globally normalized probability-utility-weighted k-NN", //6
                "Globally weighted k-NN winner take all per cluster" // 7
                },
            6);
    
    public final MultiChoiceOption positiveClusterFeedbackStrategyOption = new MultiChoiceOption("positiveClusterFeedbackStrategy", 'P',
            "Set strategy which clusters we will increase weights based on a new point's 'match'",
            new String[]{"NearestOnly", "AllClass", "AllCandidates"},
            new String[]{"Only the nearest cluster increases weight", "All clusters of that class label nearby", "Any cluster for which this data point is not an outlier to"},
            0);
    
    //public final MultiChoiceOption distanceStrategyOption = (MultiChoiceOption) VectorDistances.distanceStrategyOption.copy();
    public final MultiChoiceOption distanceStrategyOption = new MultiChoiceOption("distanceStrategy", 'd',
            "Set strategy for distance measure.",
            new String[]{"Minimum",     // 0
                "Manhattan",            // 1
                "Euclidian",            // 2
                "Chebychev",            // 3
                "Aggarwal-0.1",         // 4
                "Aggarwal-0.3",         // 5
                "Average",              // 6
                "Chord",                // 7
                "Geo",                  // 8
                "Divergence",           // 9
                "Gower",                //10
                "Bray",                 //11
                "Jaccard"},             //12
            new String[]{"Minimum distance(L0 Norm)", 
                "Manhattan distance (L1 Norm), ", 
                "Euclidian distance (L2 Norm)", 
                "Chebychev distance (L-Inf Norm)", 
                "Aggarwal L-0.1 Norm (fractional minkowski power)",
                "Aggarwal L-0.3 Norm (fractional minkowski power)",
                "Average",
                "Chord",
                "Geo",
                "Divergence",
                "Gower",
                "Bray",
                "Jaccard"},
            2);
    
    //public final MultiChoiceOption distanceStrategyOption = (MultiChoiceOption) VectorDistances.distanceStrategyOption.copy();
    public final MultiChoiceOption inclusionProbabilityStrategyOption = new MultiChoiceOption("inclusionProbabilityStrategy", 'p',
            "Set strategy for probability measure.",
            new String[]{"StdNorm",     // 0
                "StdNormPk",            // 1
                "StdNormPk_div_Px",     // 2
                "StdNormPk_div_Pxc"     // 3
                },             
            new String[]{"N(mu,sigma) : Use only standard gaussian",                          //0
                "N(mu,sigma) * P(k) : Include cluster weight",                                //1
                "N(mu,sigma) P(k) / P(x) : Bayes approach using univeral cluster for P(x)",   //2
                "N(mu,sigma) P(Ck) / P(x|C) : Bayes approach using univeral cluster for P(x)" //3
                },
            0);
    
    public final FloatOption initialStandardDeviationOption = new FloatOption("initialStandardDeviation", 'r',
            "intial StdDev for cluster upon intial creation (first data point)",
            0.05, 0.0, 10000.0);
    
    public FlagOption optimizeInitialClusterNumberOption = new FlagOption("optimizeInitialClusterNumber",'O',
                                                        "Used ICMD+MICD to optimize initial number of clusters from warmup");
    
    public final FloatOption hypothesisWeightOption = new FloatOption("clusterAdjustmentWeight", 'h',
            "For unlabeled data, contribute the cluster labelling by this amount using h(x). Note this is dangerous as h(x) drifts from g(x)",
            0.00, 0.0, 1.0);
    
    /*  From base class:
     protected InstancesHeader modelContext; // isa Instances
     protected double trainingWeightSeenByModel;
     protected int randomSeed;
     protected IntOption randomSeedOption;
     public FlagOption evaluateMicroClusteringOption;
     protected Random clustererRandom;
     protected Clustering clustering;
     */
    
    /**
     * set of all current clusters
     */
    protected Set<Riffle> clusters = new TreeSet<>();
    protected Riffle universalCluster = null;
    protected final TreeSet<Integer> knownLabels = new TreeSet<>();
    protected double defaultSigma = 0.01;
    protected ValIdxTupleType CVI = null;
    protected double universalProbabilitySums = 0;
    protected double bestProbabilitySums = 0;
    protected double bestProbabilityCount = 0;
    protected int universalOutlierCount = 0;
    protected int unanimousOutlierCount = 0;
    protected int newLabelCount = 0;
    protected int newClusterCreateCalls = 0;
    protected Instances outlierPerceptronTrainingSet = null;
    protected Perceptron[] perceptrons = null;
    protected double[]     pweights    = null;
    
    @Override
    public void resetLearningImpl() {
        this.clustering = new Clustering();
        this.modelContext = null;
        this.trainingWeightSeenByModel = 0.0;
        this.knownLabels.clear();
        if (this.universalCluster != null) {this.universalCluster.cleanTallies();}
        this.CVI = null;
    }

    @Override
    public void trainOnInstance(Instance inst) {
        this.trainingWeightSeenByModel += inst.weight();
        trainOnInstanceImpl(inst);
    }
    
    public Riffle getUniverse() {
        return this.universalCluster;
    }
    
    /**
     * Use inclusion probability to discover the cluster "nearest" the provided instance
     * @param x instance in question
     * @param C set of clusters
     * @return sorted set of clusters, ordered by probability of inclusion
     */
    protected SortedSet<NearestClusterTuple> findMostLikelyClusters(Set<Riffle> C, Instance x) {
        SortedSet<NearestClusterTuple> ret = new TreeSet<>(/*C.size()*/);
        TreeMap<Riffle,Double> temporaryProbabilities = new TreeMap<>();
        double globalProbability = 0;
        double globalWeightedProbability = 0;
        double universalProbability = this.universalCluster.getInclusionProbability(x);
        for (Riffle c : C) {
            double probability = c.getInclusionProbability(x);
            globalProbability += probability;
            globalWeightedProbability += probability * c.getWeight();
            temporaryProbabilities.put(c, probability);
        }
        
        // Gather 
        for (Riffle c : temporaryProbabilities.keySet()) {
            double pk = temporaryProbabilities.get(c) ;
            double p = pk;
            switch(this.inclusionProbabilityStrategyOption.getChosenIndex()) {
                case 0:
                    pk = p;
                    break;
                case 1:
                    pk = p * c.getWeight();
                    break;
                case 2:
                    pk = p * c.getWeight() / universalProbability;
                    break;
                case 3:
                    pk = p * c.getWeight() / globalWeightedProbability;
                    break;
                case 4: 
                    pk = p * c.getWeight() / globalProbability;
                    break;
                default:
                    pk = p;
            }
            NearestClusterTuple nct = new NearestClusterTuple(c,pk);
            ret.add(nct);
        } // end for
        return ret;
    }
    
    /**
     * Wrapper for parallel K-Means for processing warm-up data set
     * @param D Warm-up data set
     * @param K number of clusters
     * @param useLabels if true, use
     * @return 
     */
    protected Set<Riffle> batchCluster(List<Instance> D, int K, boolean useLabels) {
        assert K >= 2 : "Minimum number of clusters (K) is 2";
        int numAttributes = D.get(0).numAttributes();
        TreeSet<Riffle> ret = new TreeSet<>();
        TreeSet<Integer> labels = new TreeSet<>();
        TreeMap<Integer, TreeSet<Riffle>> potentialClusters = new TreeMap<>();
        //Create a potential cluster pool. Seperate into seperate pools by label if useLabels is set to true:
        for (Instance x : D) {
            int label = (useLabels) ? (int) x.classValue() : 0;
            labels.add(label);
            TreeSet<Riffle> clusterSet = potentialClusters.get(label);
            if (clusterSet == null) { clusterSet = new TreeSet<>(); }
            clusterSet.add(this.createNewCluster(x));
            potentialClusters.put(label, clusterSet);
        }
        
        // Initialize following the K-Means++ approach:
        Riffle C = potentialClusters.firstEntry().getValue().first();
        ret.add(C);
        potentialClusters.firstEntry().getValue().remove(C);
        
        Iterator<Integer> labelIter = labels.iterator();
        while((ret.size() < K) && !potentialClusters.isEmpty()) {
            if (!labelIter.hasNext()) { labelIter = labels.iterator(); } // loop around as needed
            int pseudoLabel = labelIter.next();
            TreeSet<Riffle> clusterSet = potentialClusters.get(pseudoLabel);
            if (clusterSet.isEmpty()) { 
                potentialClusters.remove(pseudoLabel);
                labelIter.remove();
                continue;
            }
            SortedSet<NearestClusterTuple> nearestClusters = findMostLikelyClusters(clusterSet, C.toInstance());
            C = nearestClusters.last().getCluster();
            ret.add(C);
            clusterSet.remove(C);
        }
        potentialClusters.clear();
        
        // Iterate 
        final int maxIterations = 100;
        final double minDelta = 0.0001;
        int iteration = 0;
        double valIdxDelta = 1.0;
        ValIdxTupleType lastScore = null;
        while ((iteration < maxIterations) && (valIdxDelta > minDelta)) {
            iteration++;
             ret.parallelStream().forEach((c) -> {
                c.cleanTallies();
                if (c.instances == null) {
                   c.instances = c.getHeader();
                }
                c.instances.clear();                
            });
             
            // Expectation Step
            boolean wasAdded;
            for(Instance x : D) {
                SortedSet<NearestClusterTuple> nearestClusters = findMostLikelyClusters(ret, x);
                wasAdded = false;
                int xLabel = (int) x.classValue();
                int cLabel = 0;
                if (useLabels) {
                    // Add to nearest cluster with same label
                    for(NearestClusterTuple nct : nearestClusters) {
                        cLabel = (int) nct.getCluster().getGroundTruth();
                        if (cLabel == xLabel) {
                            nct.getCluster().addInstance(x);
                            nct.getCluster().instances.add(x);
                            wasAdded = true;
                            //break;
                        }
                    }
                }
                // just add to the closest cluster
                if (!wasAdded) {
                    nearestClusters.last().getCluster().instances.add(x);
                }
            }

            // Maximization Step
            for (Riffle c : ret) {
                if (c.instances == null || c.instances.isEmpty()) {
                    continue;
                }
                double[] clusterCentroid = new double[numAttributes];
                double[] clusterVariance = new double[numAttributes];
                for (Instance x : c.instances) { // Pre-populate univeral cluster with data points
                    double[] xValues = x.toDoubleArray();
                    for (int i = 0; i < xValues.length; ++i) {
                        clusterCentroid[i] += xValues[i] / ((double) c.instances.size());
                    }
                }
                // The cluster class uses an incremental heuristic, but we want to start out as pure as possible, so
                // we use the 2-Pass method for computing sample variance (per dimension)
                if (c.instances.size() < 2) {
                    for (int i = 0; i < clusterVariance.length; ++i) {
                        clusterVariance[i] = universalCluster.getVariances()[i] * 0.85; // Statistical Variance
                    }
                } else {
                    double n = c.instances.size();
                    double[] cep = new double[numAttributes];
                    Arrays.fill(cep, 0);
                    for (Instance x : c.instances) {
                        double[] xValues = x.toDoubleArray();
                        for (int i = 0; i < xValues.length; ++i) {
                            double delta = clusterCentroid[i] - xValues[i];
                            cep[i] += delta;
                            clusterVariance[i] += delta * delta; // Statistical Variance
                        }
                    }
                    for (int i = 0; i < clusterVariance.length; ++i) {
                        clusterVariance[i] = (clusterVariance[i] - cep[i] * cep[i] / n) / (n - 1);
                    }
                }
                c.setCenter(clusterCentroid); // temporary - start with standard gaussian, gets updated below
                c.setVariances(clusterVariance);
                c.recompute(); // this updates entropies and such
//                double[] clusterCentroid = new double[numAttributes];
//                Arrays.fill(clusterCentroid, 0);
//                for (Instance x : c.instances) { // Pre-populate univeral cluster with data points
//                    double[] xValues = x.toDoubleArray();
//                    for (int i = 0; i < xValues.length; ++i) {
//                        clusterCentroid[i] += xValues[i] / ((double) c.instances.size());
//                    }
//                }
//                c.setCenter(clusterCentroid);
            }
            
            ValIdxTupleType currentScore = new ValIdxTupleType(ret);
            if (lastScore != null) {
                double diff = Math.abs(lastScore.getValIdx() - currentScore.getValIdx());
                double denominator = lastScore.getValIdx();
                valIdxDelta =  (denominator == 0) ? 0.0 : Math.abs(diff / denominator);
            }
            lastScore = currentScore;
        } // end while
        return ret;
    } // end batchCluster()

    /**
     * Uses methodology from Kim et al. "A Novel Validity Index for Determination of the Optimal Number of Clusters"
     * @param D Warm-up data set
     */
    public void initialize(List<Instance> D) {
        assert (D == null || D.isEmpty() || D.get(0) == null) : "FeS::initialize() called with a null data list!";
        knownLabels.clear();
        universalProbabilitySums = 0;
        bestProbabilitySums = 0;
        bestProbabilityCount = 0;
        // Setup the universal set/cluster. Note that this will be crucial for subspace selection (cross-entropy checks against null hypothesis)
        double[] universalCentroid = new double[D.get(0).numAttributes()];
        double[] universalVariance = new double[D.get(0).numAttributes()];
        Arrays.fill(universalCentroid, 0);
        Arrays.fill(universalVariance, 0);
        universalCluster= new Riffle(D.get(0));
        universalCluster.updateStrategyOption.setChosenIndex(this.updateStrategyOption.getChosenIndex());
        universalCluster.outlierDefinitionStrategyOption.setChosenIndex(this.outlierDefinitionStrategyOption.getChosenIndex());
        universalCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
        universalCluster.initialStandardDeviationOption.setValue(this.initialStandardDeviationOption.getValue());
        universalCluster.alphaAdjustmentWeightOption.setValue(this.learningRateAlphaOption.getValue());
        //universalCluster.setParentClusterer(this);
        if (D.size() > 1) {
            double[] ep = new double[universalCentroid.length];
            Arrays.fill(ep, 0);
            universalCluster.setCenter(universalCentroid); // temporary - start with standard gaussian, gets updated below
            universalCluster.setVariances(universalVariance); // temporary - start with standard gaussian, will update below
            universalCluster.setWeight(0);
            double N = D.size();
            for (Instance x : D) { // Pre-populate univeral cluster with data points
                knownLabels.add((int) x.classValue());
                universalCluster.addInstance(x);
                double[] xValues = x.toDoubleArray();
                for (int i = 0; i < xValues.length; ++i) {
                    universalCentroid[i] += xValues[i];
                }
            }
            for (int i = 0; i < universalCentroid.length; ++i) {
                universalCentroid[i] /= N;
            }
        // The cluster class uses an incremental heuristic, but we want to start out as pure as possible, so
            // we use the 2-Pass method for computing sample variance (per dimension)
            for (Instance x : D) {
                double[] xValues = x.toDoubleArray();
                for (int i = 0; i < xValues.length; ++i) {
                    double delta = universalCentroid[i] - xValues[i];
                    ep[i] += delta;
                    universalVariance[i] += delta * delta;
                }
            }
            for (int i = 0; i < universalVariance.length; ++i) {
                universalVariance[i] = (universalVariance[i] - ep[i] * ep[i] / N ) / (N - 1);
            }
            universalCluster.setCenter(universalCentroid); // temporary - start with standard gaussian, gets updated below
            universalCluster.setVariances(universalVariance);
        }
        universalCluster.recompute(); // this updates entropies and such
        
        // Ok, now let's use K-Means to find the initial cluster set
        int Cmin = this.clustersPerLabelOption.getValue() * this.knownLabels.size();
        int Cmax = Cmin + 1;
        if (optimizeInitialClusterNumberOption.isSet()) {
            Cmin = this.minimumNumberOfClusterSizeOption.getValue();//Math.max(knownLabels.size(), 2);
            Cmax = Math.max(Cmin + 1, Math.min(this.clustersPerLabelOption.getValue() * this.knownLabels.size(), this.maximumNumberOfClusterSizeOption.getValue()));
        }
        ArrayList<ValIdxTupleType> valIdxSet = new ArrayList<>(Cmax);
        Set<Riffle> V;
        // Create multiple hypothesis for best K choices:
        for (int c = Cmin; c < Cmax; c++) {
            V = batchCluster(D, c, true);
            ValIdxTupleType i = new ValIdxTupleType(V);
            valIdxSet.add(i);
            if (CVI == null) {
                CVI = i;
            } else {
                CVI.setVo_min(Math.min(i.getVo(), CVI.getVo_min()));
                CVI.setVo_max(Math.max(i.getVo(), CVI.getVo_max()));
                CVI.setVu_min(Math.min(i.getVu(), CVI.getVu_min()));
                CVI.setVu_max(Math.max(i.getVu(), CVI.getVu_max()));
            }
        }

        // Normalize all:
        valIdxSet.parallelStream().map((i) -> {
            i.setVo_min(CVI.getVo_min());
            return i;
        }).map((i) -> {
            i.setVo_max(CVI.getVo_max());
            return i;
        }).map((i) -> {
            i.setVu_min(CVI.getVu_min());
            return i;
        }).forEach((i) -> {
            i.setVu_max(CVI.getVu_max());
        });
        
        // Find the best K by finding the minimum score:
        valIdxSet.stream().filter((i) -> (i.getValIdx() < CVI.getValIdx())).forEach((i) -> { CVI = i; });
        
        BufferedWriter datawriter = null;                                       // DEBUG
        BufferedWriter rawdatawriter = null;                                    // DEBUG
        BufferedWriter clusterwriter = null;                                    // DEBUG
        String filePrefix = "DEBUG-" + iso8601FormatString.format(new Date());  // DEBUG
        try {                                                                   // DEBUG
            File warmupData = new File((filePrefix + "-first" + D.size() + ".csv"));     // DEBUG
            File rawwarmupData = new File((filePrefix + "-raw" + D.size() + ".csv"));    // DEBUG
            File clusterData = new File((filePrefix + "-clusters.csv"));         // DEBUG
            datawriter = new BufferedWriter(new FileWriter(warmupData));         // DEBUG
            rawdatawriter = new BufferedWriter(new FileWriter(rawwarmupData));   // DEBUG
            clusterwriter = new BufferedWriter(new FileWriter(clusterData));     // DEBUG
            clusterwriter.write("id,s,w,r,e,p,y,c,v");                             // DEBUG
            clusterwriter.newLine();                                             // DEBUG
            String csv = "";                                                     // DEBUG
            int rowCount = 0;                                                    // DEBUG
            for (Instance x : D) {                                               // DEBUG
                    double[] dataArray = x.toDoubleArray();                      // DEBUG
                    for(int dIdx = 0; dIdx < dataArray.length; ++dIdx) {         // DEBUG
                        csv += dataArray[dIdx] + ",";                            // DEBUG
                    }                                                            // DEBUG
                    csv += ++rowCount;                                           // DEBUG
                    rawdatawriter.write(csv);                                       // DEBUG
                    rawdatawriter.newLine();                                        // DEBUG
                    csv = "";                                                    // DEBUG
                }                                                                // DEBUG
            for(Double uvar : universalVariance) {
                csv += uvar + ",";
            }
            rawdatawriter.write(csv);                                       // DEBUG
            rawdatawriter.newLine();                                        // DEBUG
            csv = "";  
            for(Double umean : universalCentroid) {
                csv += umean + ",";
            }
            rawdatawriter.write(csv);                                       // DEBUG
            rawdatawriter.newLine();                                        // DEBUG
            csv = "";  
            rawdatawriter.flush();
            this.clusters.clear();
            for (Riffle c : CVI.getClustering()) {
                if (c.instances == null || c.instances.isEmpty()) {
                    continue;
                }
                double[] clusterCentroid = new double[universalCentroid.length];
                double[] clusterVariance = new double[universalVariance.length];
                for (Instance x : c.instances) { // Pre-populate univeral cluster with data points
                    double[] xValues = x.toDoubleArray();
                    for (int i = 0; i < xValues.length; ++i) {
                        clusterCentroid[i] += xValues[i] / ((double) c.instances.size());
                    }
                }
                // The cluster class uses an incremental heuristic, but we want to start out as pure as possible, so
                // we use the 2-Pass method for computing sample variance (per dimension)
                if (c.instances.size() < 2) {
                    for (int i = 0; i < clusterVariance.length; ++i) {
                        clusterVariance[i] = universalCluster.getVariances()[i] * 0.85; // Statistical Variance
                    }
                } else {
                    double n = c.instances.size();
                    double[] cep = new double[universalCentroid.length];
                    Arrays.fill(cep, 0);
                    for (Instance x : c.instances) {
                        double[] xValues = x.toDoubleArray();
                        for (int i = 0; i < xValues.length; ++i) {
                            double delta = clusterCentroid[i] - xValues[i];
                            cep[i] += delta;
                            clusterVariance[i] += delta * delta; // Statistical Variance
                        }
                    }
                    for (int i = 0; i < clusterVariance.length; ++i) {
                        clusterVariance[i] = (clusterVariance[i] - cep[i] * cep[i] / n) / (n - 1);
                    }
                }
                c.setCenter(clusterCentroid); // temporary - start with standard gaussian, gets updated below
                c.setVariances(clusterVariance);
                c.recompute(); // this updates entropies and such

                // WRITE DEBUG DATA
                
                for (Instance x : c.instances) {
                    double[] dataArray = x.toDoubleArray();
                    for(int dIdx = 0; dIdx < dataArray.length; ++dIdx) {
                        csv += dataArray[dIdx] + ",";
                    }
                    csv += c.getId();
                    datawriter.write(csv);
                    datawriter.newLine();
                    csv = "";
                }
                
                
//              clusterwriter.write("id,w,r,e,p,y,c,v");
                if (Double.isNaN(c.getRadius())) {
                    System.out.print("Bad radius");
                }
                clusterwriter.write(c.getId() + "," + c.size() + "," + c.getWeight() + "," + c.getRadius() + "," + c.getEntropy() + "," + c.getTruePurity() + "," + 
                        weka.core.Utils.maxIndex(c.getVotes()) + ",Centroid:," +
                        weka.core.Utils.arrayToString(c.getCenter()) + ",Var:," + 
                        weka.core.Utils.arrayToString(c.getVariances()));
                clusterwriter.newLine();
                // END DEBUG DATA
                
                this.clusters.add(c);
            }
            if (this.outlierDefinitionStrategyOption.getChosenIndex() == 1) {
                this.setupPerceptron();
                double outlierPerceptronTrainingError = this.trainPerceptron();
                System.out.println("outlier detection Perceptron training error = " + outlierPerceptronTrainingError);
            }
            this.clusters.stream().forEach((c) -> { c.instances.clear(); });
            this.newClusterCreateCalls = 0;
            System.out.println("Starting with " + this.clusters.size() + " clusters and " + this.knownLabels + " labels.");

            clusterwriter.flush();     // DEBUG
            clusterwriter.close();  // DEBUG
            datawriter.flush();     // DEBUG
            datawriter.close();  // DEBUG
            rawdatawriter.flush();     // DEBUG
            rawdatawriter.close();  // DEBUG
        } catch (IOException e) { } // DEBUG
} // end initialize()
    
    /**
     * In cases where this class is not used by the moa.tasks.EvaluateNonStationaryDynamicStream task, 
     * this safety (fallback) initialization procedure is necessary.
     * @param x 
     */
    public void safeInit(Instance x) {
        if( this.universalCluster == null) {
            universalCluster= new Riffle(x);
            universalCluster.updateStrategyOption.setChosenIndex(this.updateStrategyOption.getChosenIndex());
            universalCluster.outlierDefinitionStrategyOption.setChosenIndex(this.outlierDefinitionStrategyOption.getChosenIndex());
            universalCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
            universalCluster.initialStandardDeviationOption.setValue(this.initialStandardDeviationOption.getValue());
            universalCluster.alphaAdjustmentWeightOption.setValue(this.learningRateAlphaOption.getValue());
            double[] initialVariances = new double[x.numAttributes()];
            Arrays.fill(initialVariances, 1.0);
            universalCluster.setVariances(initialVariances);
            universalCluster.setWeight(0);
            universalCluster.recompute();
            this.knownLabels.clear();
            bestProbabilitySums = 0;
            bestProbabilityCount = 0;
        }
    }
    
    /**
     * 
     * @param x
     * @param c 
     */
    private void addToCluster(Instance x, Riffle c) {
        c.reward();
        c.addInstance(x);
        // If configured, use prediction as a reduced-weight training
        // WARNING: This often becomes divergant and error-inducing if done with bad weights
        double fauxWeight = this.hypothesisWeightOption.getValue();
        if ((x.weight() == 0) && (fauxWeight > 0)) { 
            int fauxClass = weka.core.Utils.maxIndex(this.getVotesForInstance(x));
            c.addLabeling(fauxClass, fauxWeight);
        }
    }
    
    /**;
     *
     * @param x instance to train on
     */
    @Override
    public void trainOnInstanceImpl(Instance x) {
        safeInit(x);
        assert (x != null) : "FeS2::trainOnInstanceImpl() Training on a null instance!";
        int classValue = (int) x.classValue();
        boolean isNewLabel = (!knownLabels.contains(classValue)) && (x.weight() > 0); 
        if ((x.weight() > 0)) { this.knownLabels.add(classValue); }
        this.universalCluster.addInstance(x);
        // Find nearest Cluster
        final SortedSet<NearestClusterTuple> nearestClusters = findMostLikelyClusters(this.clusters, x);
        assert !nearestClusters.isEmpty() : "Cluster set for probability matching is empty";
        
        // Compute some base metrics we need to know:
        double maxRadius = 0;
        double avgRadius = 0;                
        boolean unanimousOutlier = true;
        double weightTotal = 0;
        double minWeight = Double.MAX_VALUE;
        for(NearestClusterTuple nct : nearestClusters)   { 
            unanimousOutlier = unanimousOutlier && nct.getCluster().isOutlier(x);
            maxRadius = Math.max(maxRadius, nct.getCluster().getRadius());
            avgRadius += nct.getCluster().getRadius();
        }
        avgRadius /= nearestClusters.size();
        
        // Update weights
        for(NearestClusterTuple nct : nearestClusters)   { 
            Riffle c = nct.getCluster();
            c.penalize(); // unilaterally reduce weights
            int clusterMajorityClass = weka.core.Utils.maxIndex(c.getVotes());
            // increase weights for matches (define 'match' criteria by strategy parameter)
            switch(this.positiveClusterFeedbackStrategyOption.getChosenIndex()){
                case 0: // only the closest
                    if (!unanimousOutlier && c == nearestClusters.last().getCluster()) { 
                        addToCluster(x,c);
                    }
                    break;
                case 1: // All label matches
                    // This ternary condition is very important for results
                    int hypothesisClass = (x.weight() > 0) ? classValue : weka.core.Utils.maxIndex(this.getVotesForInstance(x)); 
                    if (clusterMajorityClass == hypothesisClass) {
                        addToCluster(x,c);
                    }
                    break;
                case 2: // All proximity matches
                    if (!nct.getCluster().isOutlier(x)) {
                        addToCluster(x,c);
                    }
                    break;
                default:
                    break;
            } //end switch
            weightTotal += c.getWeight();
            minWeight = Math.min(minWeight,c.getWeight());
        }
        
        // Sort by (weight / sigma)
        Riffle[] sortedClusters = new Riffle[clusters.size()];
        int i = 0;
        for(Riffle c: clusters) {
            sortedClusters[i++] = c;
        } 
        // Kuddos to Java 8 and lambda expressions for making this a one-liner:
        Arrays.parallelSort(sortedClusters, (Riffle a, Riffle b) -> Double.compare(a.getWeight() / Math.max(a.getRadius(), 1e-96), b.getWeight() / Math.max(b.getRadius(), 1e-96)));
        boolean atClusterCapacity = (this.clusters.size() >= Math.min(this.clustersPerLabelOption.getValue() * this.knownLabels.size(), this.maximumNumberOfClusterSizeOption.getValue()) );
        

        // * * *
        //
        // Results show that when average P(x|k) < Chauvenet, no new clusters, and visa versa (which is opposite of expected behavior)
        //
        // * * *
        boolean universalOutlier = this.universalCluster.isOutlier(x);
        if (isNewLabel) { newLabelCount++;}
        if (universalOutlier) { universalOutlierCount++;}
        if (unanimousOutlier) {unanimousOutlierCount++;}
        // If we have no matches at all, then the weakest clsuter is replaced by a new one with a high variance and low weight
        //if (isNewLabel || (unanimousOutlier && universalOutlier)) {   
        if (isNewLabel || unanimousOutlier) {   
            Riffle weakestLink = sortedClusters[sortedClusters.length - 1]; // get last one
            Riffle novelCluster = this.createNewCluster(x);
            //novelCluster.setRadius((avgRadius + maxRadius) / 2.0); // Set to half-way between average and max radius
            novelCluster.setWeight(weightTotal / nearestClusters.size()); // <---- Validate this ------
            weightTotal += novelCluster.getWeight(); // update for new normalization factor
            // You are the weakest link... Goodbye
            if(atClusterCapacity) { 
                weightTotal -=  weakestLink.getWeight(); // update for new normalization factor
                this.clusters.remove(weakestLink); 
            }
            // Everyone please welcome our newest contestant...
            clusters.add(novelCluster);
        }
        
        // Normalize Weights and Update variance estimates for singleton clusters
        double[] universeVariance = universalCluster.getVariances();
        double[] initialVariance = new double[universeVariance.length];
        for (int j = 0; j < initialVariance.length; ++j) {
            initialVariance[j] = universeVariance[j] * 0.85;
        }
        if (weightTotal <= 0) {
            weightTotal = 1;
        }
        for (Riffle c : this.clusters) {
            if (c.size() < 2) {
                c.setVariances(initialVariance);
            }
            c.setWeight(c.getWeight() / weightTotal);
        }
    }    
    /**
     * Find the nearest cluster, and use its most frequent label.
     * If nearest cluster has no label, then we have a novel cluster
     * Unless data point is an outlier to all clusters, then it is just an outlier
     * @param inst
     * @return 
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        assert (this.universalCluster != null) : "FeS2::getVotesForInstance() called without any initialization or training!";
        int novelClassLabel = inst.numClasses();
        int outlierLabel = novelClassLabel + 1;
        double[] votes = new double[inst.numClasses() + 2];
        if (this.clusters.isEmpty()) {
            return votes;
        }
        double[] cumulativeVotes = new double[inst.numClasses()];
        double[] cumulativeVotes_p = new double[inst.numClasses()];
        double[] cumulativeVotes_pw = new double[inst.numClasses()];
        double[] cumulativeVotes_n = new double[inst.numClasses()];
        double[] cumulativeVotes_np = new double[inst.numClasses()];
        double[] cumulativeVotes_npw = new double[inst.numClasses()];
        double[] cumulativeWinnerTakesAllVotes = new double[inst.numClasses()];
        Arrays.fill(votes, 0.0);
        Arrays.fill(cumulativeVotes, 0.0);
        Arrays.fill(cumulativeVotes_p, 0.0);
        Arrays.fill(cumulativeVotes_pw, 0.0);
        Arrays.fill(cumulativeVotes_n, 0.0);
        Arrays.fill(cumulativeVotes_np, 0.0);
        Arrays.fill(cumulativeVotes_npw, 0.0);
        Arrays.fill(cumulativeWinnerTakesAllVotes, 0.0);

        final int TRUE_CLASS = (int) inst.classValue(); // for debug watch windows only
        final SortedSet<NearestClusterTuple> nearestClusters = findMostLikelyClusters(this.clusters, inst);
        boolean memberOfAtLeastOneTrueCluster = false;
        boolean universalOutlier = true;
        double bestProbability = 0;
        double universalProbability = this.universalCluster.getInclusionProbability(inst);

        NearestClusterTuple bestMatchCluster = null;
        
        // Gather data
        for (NearestClusterTuple nct : nearestClusters) {
            double p = nct.getDistance();
            boolean localOutlier = nct.getCluster().isOutlier(inst);
            memberOfAtLeastOneTrueCluster = memberOfAtLeastOneTrueCluster || (!localOutlier && nct.getCluster().size() > this.minimumClusterSizeOption.getValue());
            universalOutlier = universalOutlier && localOutlier;
            bestProbability = Math.max(p,bestProbability);
            if (p <= 0) { continue;} 
            int localWinner = (int) nct.getCluster().getGroundTruth();
            cumulativeWinnerTakesAllVotes[localWinner] += p;
            double clusterVotes[] = nct.getCluster().getVotes();
            double clusterNormalizedVotes[] = nct.getCluster().getVotes().clone();
            if(weka.core.Utils.sum(clusterNormalizedVotes) > 0) { weka.core.Utils.normalize(clusterNormalizedVotes); }
            for (int i = 0; i < clusterVotes.length; ++i) {
                cumulativeVotes[i] += clusterVotes[i];
                cumulativeVotes_p[i] += clusterVotes[i] * p;
                cumulativeVotes_pw[i] += clusterVotes[i] * p * nct.getCluster().getWeight();
                cumulativeVotes_n[i] += clusterNormalizedVotes[i];
                cumulativeVotes_np[i] += clusterNormalizedVotes[i] * p;
                cumulativeVotes_npw[i] += clusterNormalizedVotes[i] * p * nct.getCluster().getWeight();
            }
            if (!localOutlier) {
                bestMatchCluster = nct;
            }
        } // end for

        universalProbabilitySums += universalProbability;
        bestProbabilitySums += bestProbability;
        bestProbabilityCount += 1;

        if (nearestClusters.isEmpty()) {
            votes[outlierLabel] = 1.0;
        } else {
            if(weka.core.Utils.sum(cumulativeVotes) > 0) {weka.core.Utils.normalize(cumulativeVotes);}
            if(weka.core.Utils.sum(cumulativeVotes_p) > 0) {weka.core.Utils.normalize(cumulativeVotes_p);}
            if(weka.core.Utils.sum(cumulativeVotes_pw) > 0) {weka.core.Utils.normalize(cumulativeVotes_pw);}
            if(weka.core.Utils.sum(cumulativeVotes_n) > 0) {weka.core.Utils.normalize(cumulativeVotes_n);}
            if(weka.core.Utils.sum(cumulativeVotes_np) > 0) {weka.core.Utils.normalize(cumulativeVotes_np);}
            if(weka.core.Utils.sum(cumulativeVotes_npw) > 0) {weka.core.Utils.normalize(cumulativeVotes_npw);}
            if(weka.core.Utils.sum(cumulativeWinnerTakesAllVotes) > 0) {weka.core.Utils.normalize(cumulativeWinnerTakesAllVotes);}
            switch (this.votingStrategyOption.getChosenIndex()) {
                case 0: // 1-NN - usually not the strongest
                    double[] nearestNeighborVotes = nearestClusters.last().getCluster().getVotes();
                    for (int i = 0; i < nearestNeighborVotes.length; ++i) {
                        votes[i] = nearestNeighborVotes[i];
                    }
                    break;
                case 1: // Global  k-NN - this is a poor performer
                    for (int i = 0; i < cumulativeVotes.length; ++i) {
                        votes[i] = cumulativeVotes[i];
                    }
                    break;
                case 2: // Globally probability-weighted k-NN - good, but biased towards heavy clusters
                    for (int i = 0; i < cumulativeVotes_p.length; ++i) {
                        votes[i] = cumulativeVotes_p[i];
                    }
                    break;
                case 3: // Globally probability-utility-weighted k-NN - good, but overly complex
                    for (int i = 0; i < cumulativeVotes_pw.length; ++i) {
                        votes[i] = cumulativeVotes_pw[i];
                    }
                    break;
                case 4: // Globally normalized k-NN - this is also usually a really really poor performer. Don't use it
                    for (int i = 0; i < cumulativeVotes_n.length; ++i) {
                        votes[i] = cumulativeVotes_n[i];
                    }
                    break;
                case 5: // Globally normalized probability-weighted k-NN - a safe bet
                    for (int i = 0; i < cumulativeVotes_np.length; ++i) {
                        votes[i] = cumulativeVotes_np[i];
                    }
                    break;
                case 6: // Globally normalized probability-utility-weighted k-NN - default and preferred method
                    for (int i = 0; i < cumulativeVotes_npw.length; ++i) {
                        votes[i] = cumulativeVotes_npw[i];
                    }
                    break;
                case 7: // Globally weighted k-NN winner take all per cluster - Can avoid noise, but not usually the best
                default:
                    for (int i = 0; i < cumulativeWinnerTakesAllVotes.length; ++i) {
                        votes[i] = cumulativeWinnerTakesAllVotes[i];
                    }
            } // end switch
            double voteAccumulator = 0;
            for(double v : votes) {
                voteAccumulator += v;
            }
            // A novel cluster is one of sufficient size but no label
            if ((bestMatchCluster != null) // It matches a cluster
                    && (bestMatchCluster.getCluster().size() > this.minimumClusterSizeOption.getValue())  // that is overall large enough
                    && (bestMatchCluster.getCluster().getNumLabeledPoints() < 1)){ // but without labels
                votes[novelClassLabel] = 1.0;
            } 
            // outlier detection
            if (universalOutlier) {
                int maxIdx = weka.core.Utils.maxIndex(votes);
                if (maxIdx < 0) { maxIdx = 0; }
                double outlierValue = votes[maxIdx];
                if (outlierValue <= 0) {
                    votes[novelClassLabel] = 1.0; // special case of novelty when we have absolutely no clue how to label an outlier
                    outlierValue = 1e-16;
                }
                votes[outlierLabel] = outlierValue / 2.0; //Math.max(Math.abs(1.0 - bestProbability), Math.abs(1.0 - universalProbability));
            }
        } // end if (nearestClusters not empty)
        return votes;
    } // end getVotesForInstance()

    
    /**
     * Wraps new cluster creation steps
     * @param exemplar
     * @return newly created cluster
     */
    protected Riffle createNewCluster(Instance exemplar) {
            Riffle newCluster = new Riffle(exemplar);
            newCluster.updateStrategyOption.setChosenIndex(this.updateStrategyOption.getChosenIndex());
            newCluster.outlierDefinitionStrategyOption.setChosenIndex(this.outlierDefinitionStrategyOption.getChosenIndex());
            //newCluster.distanceNormStrategyOption.setChosenIndex(this.distanceNormStrategyOption.getChosenIndex());
            newCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
            newCluster.initialStandardDeviationOption.setValue(this.initialStandardDeviationOption.getValue());
            newCluster.alphaAdjustmentWeightOption.setValue(this.learningRateAlphaOption.getValue());
            newCluster.setWeight(this.initialClusterWeightOption.getValue());
            newCluster.setRadius(this.initialStandardDeviationOption.getValue());
            double[] universeVariance = universalCluster.getVariances();
            double[] initialVariance = new double[universeVariance.length];
            for(int i = 0; i < initialVariance.length; ++i) {
                initialVariance[i] = universeVariance[i] * 0.85;
            }
            newCluster.setVariances(initialVariance);
            //newCluster.setParentClusterer(this);
            newCluster.recompute();
            newClusterCreateCalls++;
            return newCluster;
    }
    
    
    /**
     * Use configurable strategy for subspace selection to re-weight the attributes of each cluster
     */
    protected void updateSubspaceSelectionWeights() {
        switch (this.subspaceStrategyOption.getChosenIndex()) {
            case 0: //none
                break;
            case 1: // K-L Divergence
                updateSubspaceSelectionWeightsViaKLDivergence();
                break;
            case 2: // K-L Divergence
                updateSubspaceSelectionWeightsViaKLNorm();
                break;
            case 3: // Variance
                updateSubspaceSelectionWeightsViaNormalizedVariance();
                break;
            case 4: // Info-Gain
                updateSubspaceSelectionWeightsViaGain();
                break;
            case 5: // Random
                updateSubspaceSelectionWeightsViaRandom();
                break;
            default:
                break;
        }
    }
    
    /**
     * Use KL-Divergence subspace selection strategy to re-weight the attributes of each cluster
     * As much as MatLab-style code annoys me in Java/C++ code, this formula setup is complex enough to merit it here
     * at least until it is all debugged - then it can be made slightly more efficient.
     */
    protected void updateSubspaceSelectionWeightsViaKLDivergence() {
        double[] baseMeans = this.universalCluster.getCenter();
        double[] baseVariances = this.universalCluster.getVariances();

        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        this.clusters.parallelStream().forEach((Riffle c) -> {
            double[] clusterVariances = c.getVariances();
            double[] clusterMeans = c.getCenter();
            double[] KLDistances = new double[baseVariances.length];
            double[] sortedKLDistances = new double[KLDistances.length];
            // Compute the K-L Divergance metric for each attribute independantly
            for (int i = 0; i < KLDistances.length && i < clusterMeans.length; ++i) {
                double DMean = baseMeans[i];
                double DVar = baseVariances[i];
                double cMean = clusterMeans[i];
                double cVar = clusterVariances[i];
                double term1 = Math.log(cVar / DVar);
                double term2 = DVar / cVar;
                double term3 = (DMean - cMean) * (DMean - cMean);
                if (Double.isNaN(term1)) {
                    term1 = 0;
                }
                if (Double.isNaN(term2)) {
                    term2 = 0;
                }
                if (Double.isNaN(term3)) {
                    term3 = 0;
                }
                double KLDist = 0.5 * (term1 + term2 + term3 - 1);
                KLDistances[i] = KLDist;
                sortedKLDistances[i] = KLDist;
            } // end for(attributes)
            parallelSort(sortedKLDistances);

            
            //Find knee of curve
            double x1 = 0.0;
            double y1 = sortedKLDistances[0];
            double xn = sortedKLDistances.length;
            double yn = sortedKLDistances[sortedKLDistances.length - 1];
            double m = (yn - y1) / (xn - x1);
            double b = yn - (m * xn);
            double maxDistanceToLine = 0.0;
            double threshold = sortedKLDistances[(int) Math.floor(sortedKLDistances.length / 2.0)];
            for (int i = 0; i < sortedKLDistances.length; ++i) {
                double currentDistanceToLine = Math.abs((m * i + b) - sortedKLDistances[i]);
                if (Double.isFinite(currentDistanceToLine) && currentDistanceToLine >= maxDistanceToLine) {
                    maxDistanceToLine = currentDistanceToLine;
                    threshold = sortedKLDistances[i];
                }
            }
            double[] newWeights = new double[KLDistances.length];
            for(int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = (KLDistances[i] <= threshold) ? 1 : 0;
            }
            c.setAttributeWeights(newWeights);
        });
    } //end updateSubspaceSelectionWeightsViaKLDivergence()

    /**
     * Use KL-Divergence Normalized Weighing strategy to re-weight the attributes of each cluster
     * As much as MatLab-style code annoys me in Java/C++ code, this formula setup is complex enough to merit it here
     * at least until it is all debugged - then it can be made slightly more efficient.
     */
    protected void updateSubspaceSelectionWeightsViaKLNorm() {
        double[] baseMeans = this.universalCluster.getCenter();
        double[] baseVariances = this.universalCluster.getVariances();

        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        this.clusters.parallelStream().forEach((Riffle c) -> {
            double[] clusterVariances = c.getVariances();
            double[] clusterMeans = c.getCenter();
            double[] KLDistances = new double[baseVariances.length];
            // Compute the K-L Divergance metric for each attribute independantly
            for (int i = 0; i < KLDistances.length && i < clusterMeans.length; ++i) {
                double DMean = baseMeans[i];
                double DVar = Math.max(baseVariances[i], weka.core.Utils.SMALL);
                double cMean = clusterMeans[i];
                double cVar = Math.max(clusterVariances[i], weka.core.Utils.SMALL);
                KLDistances[i] = Math.max(VectorDistances.KLDiverganceGaussian(cMean, cVar, DMean, DVar),weka.core.Utils.SMALL);
            } // end for(attributes)

            weka.core.Utils.normalize(KLDistances);
            c.setAttributeWeights(KLDistances);
        });
    } // end updateSubspaceSelectionWeightsViaKLNorm()
    
     /**
     * Use Variance subspace selection strategy to re-weight the attributes of each cluster
     * Do not expect this option to gain much, but it is here for comparative testing.
     */
    protected void updateSubspaceSelectionWeightsViaNormalizedVariance() {
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        this.clusters.parallelStream().forEach((Riffle c) -> {
            double[] clusterVariances = c.getVariances();
            
            double[] newWeights = new double[clusterVariances.length];
            for(int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = Math.min(clusterVariances[i],weka.core.Utils.SMALL);
            }
            weka.core.Utils.normalize(newWeights);
            c.setAttributeWeights(newWeights);
        });
    }
    
    /**
     * Use Gain-like construct for subspace selection strategy to re-weight the attributes of each cluster
     */
    protected void updateSubspaceSelectionWeightsViaGain() {
        double[] baseEntropies = this.universalCluster.getEntropies();
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        this.clusters.parallelStream().forEach((Riffle c) -> {
            double[] clusterEntropies = c.getEntropies();
            double[] newWeights = new double[clusterEntropies.length];
            // Compute the K-L Divergance metric for each attribute independantly
            double EntropyTotal = 0;
            for (int i = 0; i < baseEntropies.length && i < clusterEntropies.length; ++i) {
                double Hu = baseEntropies[i];
                double Hc = clusterEntropies[i];
                double diff = (Hu == 0) ? 0 : (Hu - Hc) / Hu;
                if (Double.isNaN(diff)) { diff = weka.core.Utils.SMALL; }
                EntropyTotal += diff;
                newWeights[i] = diff;
            } // end for(attributes)
            if (EntropyTotal == 0) { EntropyTotal = 1; }
            for(int i =0; i < newWeights.length; ++i) {
                newWeights[i] /= EntropyTotal;
            }
            c.setAttributeWeights(newWeights);
        });
    }
    
    /**
     * Use Variance subspace selection strategy to re-weight the attributes of each cluster
     * Do not expect this option to gain anything, but it is here for comparative testing (and sanity checks).
     */
    protected void updateSubspaceSelectionWeightsViaRandom() {
        int numAttribs = this.universalCluster.getCenter().length;
        Random rng = new Random(numAttribs);
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        this.clusters.parallelStream().forEach((Riffle c) -> {
            
            double[] newWeights = new double[numAttribs];
            for(int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = rng.nextDouble();
            }
            weka.core.Utils.normalize(newWeights);
            c.setAttributeWeights(newWeights);
        });
    }
    
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        double avgChLmt = 0;
        avgChLmt = this.clusters.stream().map((c) -> c.getChauvenetLimit() / this.clusters.size()).reduce(avgChLmt, (accumulator, _item) -> accumulator + _item);
        Measurement[] ret = new Measurement[]{
            new Measurement("NumClusters",
            this.clusters.size()),
            new Measurement("#NewClusters",
            this.newClusterCreateCalls),
            new Measurement("AvgClusterSize",
            this.getAverageSize()),
            new Measurement("AvgClusterLabels",
            this.getAverageLabels()),
            new Measurement("AvgClusterRadius",
            this.getAverageVariance()),
            new Measurement("UniverseRadius",
            this.universalCluster.getRadius()),
            new Measurement("WeightTotalSeen",
            this.trainingWeightSeenByModel),
            new Measurement("ClusterTruePurity",
            this.getAverageTruePurity()),
            new Measurement("ClusterTrueEntropy",
            this.getAverageTrueEntropy()),
            new Measurement("ClusterPurity",
            this.getAveragePurity()),
            new Measurement("ClusterEntropy",
            this.getAverageEntropy()),
            new Measurement("ClusteringKnownLabels",
            this.knownLabels.size()),
            new Measurement("varianceRatio",
            this.getAverageVariance() / Math.max(weka.core.Utils.SMALL, this.universalCluster.getRadius())),
            new Measurement("AvgBestMatchProb",
            (bestProbabilityCount > 0) ? bestProbabilitySums / bestProbabilityCount : -1),
            new Measurement("AvgChauvenetLimit",
            avgChLmt),
            new Measurement("AvgUniversalProb",
            (bestProbabilityCount > 0) ? universalProbabilitySums / bestProbabilityCount : -1),
            new Measurement("UniversalChauvenet",
            this.universalCluster.getChauvenetLimit()),
            new Measurement("NewLabelCounter",
            this.newLabelCount),
            new Measurement("UnivOutlierCounter",
            this.universalOutlierCount),
            new Measurement("UnanOutlierCounter",
            this.unanimousOutlierCount)
        // TODO - there are other measurements we probably want...
        };
        universalProbabilitySums = 0;
        bestProbabilitySums = 0;
        bestProbabilityCount = 0;
        newClusterCreateCalls = 0;
        newLabelCount = 0;
        universalOutlierCount = 0;
        unanimousOutlierCount = 0;
        return ret;
    }

    /**
     * 
     * @return 
     */
    protected double getAverageTruePurity() {
        if (this.clusters.isEmpty()) { return 1;}
       double ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getTruePurity()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       ret /= this.clusters.size();
       return ret;
    }
    
    /**
     * 
     * @return 
     */
    protected double getAverageTrueEntropy() {
        if (this.clusters.isEmpty()) { return 1;}
       double ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getTrueEntropy()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       ret /= this.clusters.size();
       return ret;
    }
    
       /**
     * 
     * @return 
     */
    protected double getAveragePurity() {
        if (this.clusters.isEmpty()) { return 1;}
       double ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getPurity()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       ret /= this.clusters.size();
       return ret;
    }
    
    /**
     * 
     * @return 
     */
    protected double getAverageEntropy() {
        if (this.clusters.isEmpty()) { return 1;}
       double ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getEntropy()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       ret /= this.clusters.size();
       return ret;
    }
    
     /**
     * 
     * @return 
     */
    protected double getAverageVariance() {
        if (this.clusters.isEmpty()) { return 1;}
       double ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getRadius()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       ret /= this.clusters.size();
       return ret;
    }
    
        /**
     * 
     * @return 
     */
    protected double getAverageSize() {
       if (this.clusters.isEmpty()) { return 0;}
       int ret = 0;
       ret = clusters.parallelStream().map((c) -> c.size()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       return  ((double)ret) / ((double)this.clusters.size());
    }
    
    /**
     * 
     * @return 
     */
    protected double getAverageLabels() {
       if (this.clusters.isEmpty()) { return 0;}
       int ret = 0;
       ret = clusters.parallelStream().map((c) -> c.getNumLabeledPoints()).reduce(ret, (accumulator, _item) -> accumulator + _item);
       return ((double)ret) / ((double)this.clusters.size());
    }
    
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "'Find Entities in SubSpace (FeS2) using " + this.clusters.size() + " clusters.");
        StringUtils.appendNewline(out);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public Clustering getClusteringResult() {
        this.clustering = new Clustering();
        this.clusters.stream().forEach((c) -> {
            clustering.add(c);
        });
        return this.clustering;
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
            ret = this.clusters.stream().map((c) -> c.measureByteSize()).reduce(ret, Integer::sum);
            ret += this.universalCluster.measureByteSize();
            ret += 84;
        }
        return ret;
    }
 
    /**
     * Initialized the perceptron that learns to detect outliers
     */
    protected void setupPerceptron() {
        ArrayList<String> labels = new ArrayList<>(2);
        labels.add("Member");
        labels.add("Outlier");
        
        ArrayList<Attribute> attribs = new ArrayList<>(7);
        attribs.add(new Attribute("P"));    // 0
        attribs.add(new Attribute("D"));    // 1
        attribs.add(new Attribute("PxD"));  // 2
        attribs.add(new Attribute("Chauv"));  // 3
        attribs.add(new Attribute("isOutlier",labels)); // 4
               
        attribs.stream().forEach((a) -> { a.setWeight(1.0); });
        attribs.get(attribs.size() - 1).setWeight(0);
        outlierPerceptronTrainingSet = new Instances("PerceptronTrainer",attribs, 5000 * this.clusters.size());
        outlierPerceptronTrainingSet.setClassIndex(outlierPerceptronTrainingSet.numAttributes() - 1); //zero-indexed so last
        outlierPerceptronTrainingSet.setClass(attribs.get(attribs.size() - 1));
    }
    
    /**
     * 
     * @param c cluster that is being compared against
     * @param x real data instance
     * @return DenseInstance made to work with the outlier-detecting perceptron
     */
    private Instance makePerceptronInstance(Riffle c, Instance x) {
        Instance pseudoPoint = new DenseInstance(this.outlierPerceptronTrainingSet.numAttributes());
         pseudoPoint.setDataset(outlierPerceptronTrainingSet);
         double p = c.getInclusionProbability(x);
         double r = (c.getRadius() != 0 ) ? c.getRadius() : 1;
         //double w = c.getWeight();
         double N =  Math.min(c.size(), 1.0 / (this.learningRateAlphaOption.getValue() + 1e-9));
         double d = c.getCenterDistance(x);
         double logP = (p == 0) ? 0 : Math.log(p);
         double logDR = (r == 0 || (d / r) == 0) ? 0 : Math.log(d / r);
         pseudoPoint.setValue(0, logP);
         pseudoPoint.setValue(1, logDR);
         pseudoPoint.setValue(2, logDR * logP);
         pseudoPoint.setValue(3, logP - Math.log(1.0 / Math.pow(2.0 * N, this.universalCluster.getHeader().numAttributes())));
         pseudoPoint.setClassValue(0);
         pseudoPoint.setWeight(0.0);
         return pseudoPoint;
    }
    
    /**
     * @return training accuracy
     */
    private double trainPerceptron() {
        // Train the perceptron from warmup phase clustering 
        final int epochs = 20;
        final int numberOfPerceptrons = 10;
        final int MEMBER  = 0;
        final int OUTLIER = 1;
        double accuracySum = 0;
        double accuracyCount = 0;
        this.outlierPerceptronTrainingSet.clear();
        Random rng = new Random(this.randomSeed);
        
        // Generate training set
        for (Riffle thisCluster : this.clusters) {
            for (Riffle thatCluster : this.clusters) {
                double groundTruth = (thisCluster == thatCluster) ? MEMBER : OUTLIER;
                for (Instance x : thatCluster.getHeader()) {
                    Instance pseudoPt = makePerceptronInstance(thisCluster, x);
                    pseudoPt.setClassValue(groundTruth);
                    this.outlierPerceptronTrainingSet.add(pseudoPt);
                }
            }
        }
        this.outlierPerceptronTrainingSet.parallelStream().forEach((x) -> {
            x.setWeight(1.0 / this.outlierPerceptronTrainingSet.numInstances());
        });
        
        // Boost it
        this.perceptrons = new Perceptron[numberOfPerceptrons];
        this.pweights = new double[numberOfPerceptrons];
        for (int perceptronIdx = 0; perceptronIdx < numberOfPerceptrons; ++perceptronIdx) {
            // Discover new weak learner
            Perceptron candidatePerceptron = new Perceptron();
            candidatePerceptron.prepareForUse();
            candidatePerceptron.learningRatioOption.setValue(rng.nextDouble() * 0.9 + 0.1);
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (Instance x : this.outlierPerceptronTrainingSet) {
                    if ((rng.nextDouble() / this.outlierPerceptronTrainingSet.numInstances()) < x.weight()) { // weighted subsampling
                        candidatePerceptron.trainOnInstance(x);
                    }
                }
            } //end epochs
            // Evaluate weak learner
            double errorFunctionSum = 0;
            double weightSum = 0;
            for (Instance x : this.outlierPerceptronTrainingSet) {
                if (!candidatePerceptron.correctlyClassifies(x)) { 
                    errorFunctionSum += x.weight();
                }
            }
             // adjust training weights
            for (Instance x : this.outlierPerceptronTrainingSet) {
                double newWeight = x.weight();
                if (candidatePerceptron.correctlyClassifies(x)) { 
                    newWeight *= errorFunctionSum / (1.0 - errorFunctionSum);
                    if (Double.isNaN(newWeight)){
                        newWeight = weka.core.Utils.SMALL;
                    }
                    x.setWeight(newWeight);
                }
                weightSum += newWeight;
            }
            // Normalize
            for (Instance x : this.outlierPerceptronTrainingSet) {
               x.setWeight(x.weight() / weightSum);
            }
            // Add to ensemble
            double newPerceptronWeight = Math.log((1 - errorFunctionSum) / errorFunctionSum);
            
            this.perceptrons[perceptronIdx] = candidatePerceptron;
            this.pweights[perceptronIdx] = newPerceptronWeight;
        } // end numPerceptrons

        // Check training error
        accuracySum = 0;
        accuracyCount = 0;
        for (Instance x : this.outlierPerceptronTrainingSet) {
            if (this.getPerceptronVotesForOutlierStatus(x) == x.classValue()) {
                accuracySum++;
            }
            accuracyCount++;
        }
        double trainingAccuracy = (accuracyCount > 0) ? (accuracySum / accuracyCount) : 0.0;
        this.outlierPerceptronTrainingSet.clear();
        return trainingAccuracy;
    }

    /**
     * 
     * @param x pseudoPoint
     * @return hypthesized class for x
     */
    public double getPerceptronVotesForOutlierStatus(Instance x) {
        assert this.perceptrons != null : "Perceptron list is not yet initialized";
        double votes[] = new double[2];
        double voteSum = 0;
        for(int i = 0; i < this.perceptrons.length && i < this.pweights.length; ++i) {
            double localVotes[] = this.perceptrons[i].getVotesForInstance(x);
            for(int v = 0; v < localVotes.length && v < votes.length; ++v) {
                double delta = this.pweights[i] * localVotes[v];
                votes[v] +=  (Double.isNaN(delta)) ? 0 : delta;
                voteSum += (Double.isNaN(delta)) ? 0 : delta;
            }
        }
        if (voteSum != 0) { weka.core.Utils.normalize(votes); }
        return weka.core.Utils.maxIndex(votes);
    }
        
    /**
     * 
     * @param cluster cluster that is being compared against
     * @param x Instance to compare with the cluster
     * @return true if x is an outlier to the cluster given the attributes of cluster and probability p
     */
    public boolean askThePerceptronIfImAnOutlier(Riffle cluster, Instance x) {
         Instance pseudoInstance = makePerceptronInstance(cluster, x);
         return ( 1 == getPerceptronVotesForOutlierStatus(pseudoInstance));
    }
    
}

