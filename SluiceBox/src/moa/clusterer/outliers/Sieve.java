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
package moa.clusterer.outliers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import static java.util.Arrays.parallelSort;
import static java.util.Arrays.parallelSort;
import static java.util.Arrays.parallelSort;
import static java.util.Arrays.parallelSort;
import java.util.Collection;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import moa.classifiers.Classifier;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.cluster.Clustering;
import moa.cluster.Riffle;
import moa.clusterers.outliers.MyBaseOutlierDetector;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.core.VectorDistances;
import moa.core.tuples.NearestClusterTuple;
import moa.core.tuples.NearestInstanceTuple;
import moa.core.tuples.ValIdxTupleType;
import moa.options.ClassOption;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * SIEVE - Streaming Incremental Expectation-Maximization in Volatile Environments This class was originally designed for
 * use as part of Brandon Parker's Dissertation work.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 3 $
 */
final public class Sieve extends MyBaseOutlierDetector /*AbstractClusterer*/ {

    private static final long serialVersionUID = 1L;
    protected static final DateFormat iso8601FormatString = new SimpleDateFormat("yyyyMMdd'T'HHmmssSSS");

    public final IntOption minimumClusterSizeOption = new IntOption("minimumClusterSize", 'm',
            "Minimum size for a cluster to be used as a labeled cluster",
            15, 1, Integer.MAX_VALUE);

    public final IntOption minimumNumberOfClusterSizeOption = new IntOption("minNumClusters", 'k',
            "Minimum number of clusters to keep",
           10, 1, Integer.MAX_VALUE);

    public final IntOption maximumNumberOfClusterSizeOption = new IntOption("maxNumClusters", 'K',
            "Minimum number of clusters to keep",
            75, 2, Integer.MAX_VALUE);

    public final IntOption clustersPerLabelOption = new IntOption("ClustersPerLabel", 'g',
            "Preferred number of clusters to keep per known label, subject to min/max num cluster constraints",
            5, 1, Integer.MAX_VALUE);

    public final IntOption cacheSizeOption = new IntOption("cacheSize", 'c',
            "Small data cache to retain for clustering correction",
            4000, 1, Integer.MAX_VALUE);

    public final IntOption loopsPerIterationOption = new IntOption("loopsPerIteration", 'i',
            "Minimum number of clusters to keep",
            20, 1, 100);

    public final IntOption resynchIntervalOption = new IntOption("resynchInterval", 'R',
            "How often to resynch incremental clustering method with cached values",
            250, 1, Integer.MAX_VALUE);

    public final MultiChoiceOption votingStrategyOption = new MultiChoiceOption("votingStrategy", 'V',
            "Set strategy for tabulating voting for predicted label among the online clusters (i.e. 'neighbor' exemplars).",
            new String[]{"1-NN", "G-NN", "GW-NN", "GWH-NN", "GN-NN", "GNP-NN", "GNWH-NN", "GWS-NN"},
            new String[]{"1-NN", // 0
                "Global  k-NN", // 1
                "Globally distance-weighted k-NN", // 2    
                "Globally distance-utility-entropy k-NN", // 3
                "Globally normalized k-NN", // 4
                "Globally normalized distance-weighted k-NN", // 5
                "Globally normalized entropy-distance-weighted k-NN", //6
                "Globally weighted k-NN winner take all per cluster" // 7
        },
            1);

    public final MultiChoiceOption positiveClusterFeedbackStrategyOption = new MultiChoiceOption("positiveClusterFeedbackStrategy", 'P',
            "Set strategy which clusters we will increase weights based on a new point's 'match'",
            new String[]{"NearestOnly", "NearestMatchingLabel", "NearestPartialMatch"},
            new String[]{"Add to nearest cluster", "Add to Nearest clsuter that has the same dominant label", "Add to nearest that has a non-zero affinity for label"},
            1);

    public final MultiChoiceOption distanceStrategyOption = new MultiChoiceOption("distanceStrategy", 'd',
            "Set strategy for distance measure.",
            new String[]{"Minimum", // 0
                "Manhattan", // 1
                "Euclidian", // 2
                "Chebychev", // 3
                "Aggarwal-0.1", // 4
                "Aggarwal-0.3", // 5
                "Average", // 6
                "Chord", // 7
                "Geo", // 8
                "Divergence", // 9
                "Gower", //10
                "Bray", //11
                "Jaccard", //12
                "Probability"}, //13
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
                "Jaccard",
                "P(x | c)"},
            10);

    public final MultiChoiceOption inclusionProbabilityStrategyOption = new MultiChoiceOption("inclusionProbabilityStrategy", 'p',
            "Set strategy for probability measure, if probability is used as distance above",
            new String[]{"StdNorm", // 0
                "StdNormPk", // 1
                "StdNormPk_div_Px", // 2
                "StdNormPk_div_Pxc" // 3
        },
            new String[]{"N(mu,sigma) : Use only standard gaussian", //0
                "N(mu,sigma) * P(k) : Include cluster weight", //1
                "N(mu,sigma) P(k) / P(x) : Bayes approach using univeral cluster for P(x)", //2
                "N(mu,sigma) P(Ck) / P(x|C) : Bayes approach using univeral cluster for P(x)" //3
        },
            0);

    public final FlagOption optimizeInitialClusterNumberOption = new FlagOption("optimizeInitialClusterNumber", 'O',
            "Used ICMD+MICD to optimize initial number of clusters from warmup");

    public final MultiChoiceOption subspaceStrategyOption = new MultiChoiceOption("subspaceStrategy", 's',
            "Set strategy subspace selection.",
            new String[]{"none", "K-L Curve", "K-L Norm", "Variance", "Info Gain Rank", "random"},
            new String[]{"none", "Keep only attributes right of the Curve of ordered K-L divergances", "Normalize weigh by K-L Ranking", "Normalized Variance ranking", "Normalized rank based on attribute information gain", "Choose Random Weights"},
            4);

    public final FlagOption usePYForEMOption = new FlagOption("dontUsePYForEM", 'y', "Unless checked, Scale distance by P(y) for M-Step of E-M");
    public final FlagOption novelIfOutlierOption = new FlagOption("novelIfOutlier", 'F', "Novel Condition is fulfilled if x is an outlier to nearest cluster");
    
    public final FlagOption onlyCreateNewClusterAtResyncOption = new FlagOption("onlyCreateNewClusterAtResync", 'r', "If set, new cluster generation will only occur at resynch time (versus on-the-fly)");
    public final FlagOption logMetaRecordsOption = new FlagOption("logMetaRecords", 'Z', "Creates CSV of meta-data for algorithm analysis");
    public final FlagOption useqNSCAtTestOption = new FlagOption("useqNSCAtTest", 'Q', "Do final one last q-NSC(x) test in addition to cluster test before finalizing votes");
    
    //public final FlagOption novelIfTooFewVotesOption = new FlagOption("novelIfTooFewVotes", 'F', "Novel Condition is fulfilled if cluster labels are too few");
    //public final FlagOption novelIfTooSmallMaxVoteOption = new FlagOption("novelIfTooSmallMaxVote",'v',"Novel Condition is fulfilled if max voted label is too small");

    //    public final FloatOption initialStandardDeviationOption = new FloatOption("initialStandardDeviation", 'r',
//            "intial StdDev for cluster upon intial creation (first data point)",
//            0.05, 0.0, 10000.0);
    //    public final FloatOption initialClusterWeightOption = new FloatOption("initialClusterWeight", 'w',
//            "Weight to set for cluster upon intial creation (first data point)",
//            1.0, 0.0, 1.0);
//    public final FloatOption pruneThresholdOption = new FloatOption("pruneThreshold", 'e',
//            "Minimum weight a cluster can have before it is pruned",
//            0.001, 0.0, 1.0);
    public final ClassOption embeddedLearnerOption = new ClassOption("embeddedLearner", 'E',
            "Classifier for cluster label hypothesis", Classifier.class, "bayes.NaiveBayes");

    public final MultiChoiceOption outlierDefinitionStrategyOption = new MultiChoiceOption("outlierDefinitionStrategy", 'o',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Chauvenet", "EmbeddedLearner", "2.5-sigma", "3-sigma", "6-sigma", "oracle"},
            new String[]{"Chauvenet", "Used trained classifer", "2.5-sigma", "3-sigma", "6-sigma", "Cheat and use ground truth (for unit testing purposes)"},
            1);

//    public final MultiChoiceOption distanceNormStrategyOption = new MultiChoiceOption("distanceNormStrategy", 'n',
//            "Set strategy for attribute normalization for distances.",
//            new String[]{"none", "weight", "variance", "weight/variance", "random"},
//            new String[]{"none", "weight", "variance", "weight/variance", "random"},
//            0);
    public final MultiChoiceOption updateStrategyOption = new MultiChoiceOption("UpdateStrategy", 'u',
            "Set strategy for cluster updates when point is added or removed.",
            new String[]{"Stauffer-Grimson", "Shephard", "cache"},
            new String[]{"Gaussian update based on Stauffer and Grimson (1999)", "Robust update to momemts based on Shephard (1996)", "cache and compute"},
            1);
    /**
     * set of all current clusters
     */
    protected LinkedList<Riffle> clusters = new LinkedList<>();
    protected Riffle universalCluster = null;
    protected int knownLabels[] = null;
    protected int numAttributes = 0;
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
    protected double[] pweights = null;
    protected int resyncDelayCount = 0;
    protected double instancesSeen =0;
    protected double weightsSeen = 0;

    final protected class ClusterPointPair {

        public Instance x;
        public Riffle c;
        private double xArray[] = null;

        public ClusterPointPair(Instance inst, Riffle cluster) {
            x = inst;
            c = cluster;
            xArray = inst.toDoubleArray();
        }

        public double[] toDoubleArray() {
            if (xArray == null) {
                xArray = x.toDoubleArray();
            }
            return xArray;
        }
    }
    protected LinkedList<ClusterPointPair> hopperCache = new LinkedList<>();
    // This is an insert-ordered set (queue) that also provides amortized O(1) access to a member
    protected LinkedHashSet<Instance> potentialNovels = new LinkedHashSet<>(cacheSizeOption.getValue(), 0.95f);
    protected Instances header = null;

    @Override
    public void resetLearningImpl() {
        this.clustering = new Clustering();
        this.modelContext = null;
        this.trainingWeightSeenByModel = 0.0;
        if (this.knownLabels != null) {
            Arrays.fill(knownLabels, 0);
        }
        if (this.universalCluster != null) {
            this.universalCluster.cleanTallies();
        }
        this.CVI = null;
        resyncDelayCount = 0;
    }

    @Override
    public final void trainOnInstance(Instance inst) {
        this.trainingWeightSeenByModel += inst.weight();
        trainOnInstanceImpl(inst);
    }

    public Riffle getUniverse() {
        return this.universalCluster;
    }

    /**
     * Use inclusion probability to discover the cluster "nearest" the provided instance
     *
     * @param x instance in question
     * @param C set of clusters
     * @return sorted set of clusters, ordered by inc
     */
    protected final NearestClusterTuple[] findMostLikelyClusters(Collection<Riffle> C, Instance x) {
        NearestClusterTuple[] ret = new NearestClusterTuple[C.size()];
        double[] xVals = x.toDoubleArray();
        int idx = 0;
        double dist = 0;
        for (Riffle c : C) {
            dist =  c.getCenterDistance(xVals);
            ret[idx++] = new NearestClusterTuple(c, dist);
        } // end for
        Arrays.parallelSort(ret);
        return ret;
    }

    /**
     * Use inclusion probability to discover the cluster "nearest" the provided instance
     *
     * @param D instance set to sort from
     * @param x instance in question
     * @return sorted set of clusters, ordered by inc
     */
    protected final NearestInstanceTuple[] findNearestNeighbors(Instances D, Instance x) {
        NearestInstanceTuple[] ret = new NearestInstanceTuple[D.size()];
        double[] xVals = x.toDoubleArray();
        int idx = 0;
        for (Instance n : D) {
            ret[idx++] = new NearestInstanceTuple(n, VectorDistances.distance(xVals, n.toDoubleArray(), D, this.distanceStrategyOption.getChosenIndex()));
        } // end for
        Arrays.parallelSort(ret);
        return ret;
    }
    
    /**
     * Use inclusion probability to discover the cluster "nearest" the provided instance
     * Uses main object's outlier container
     * @param x instance in question
     * @return sorted set of clusters, ordered by inc
     */
    protected final NearestInstanceTuple[] findNearestOutliers(Instance x) {
        NearestInstanceTuple[] ret = new NearestInstanceTuple[potentialNovels.size()];
        double[] xVals = x.toDoubleArray();
        int idx = 0;
        for (Instance n : potentialNovels) {
            double distance = VectorDistances.distance(xVals, n.toDoubleArray(), x.dataset(), this.distanceStrategyOption.getChosenIndex());
            NearestInstanceTuple nit = new NearestInstanceTuple(n, distance);
            ret[idx++] = nit;
        } // end for
        Arrays.parallelSort(ret);
        return ret;
    }
    
    
    /**
     * This is not your grandpa's E-M algorithm... it has multiple mini-steps,
     * but "The e1-m1-e2-m2-e3-m3-Algorithm" is a mouthful, so we just call it *-Means Clustering
     * {Pronounced "Any-means (necessary) clustering"}
     * @param D
     * @param subclusters
     * @param maxK
     * @return score at the end of the process
     */
    protected final double EMStep(List<ClusterPointPair> D, Collection<Riffle> subclusters, int maxK) {
        double ret = 0;
        // clear the pallette
        for (Riffle c : subclusters) {
            if (c.instances == null) {
                c.instances = c.getHeader();
            }
            c.instances.clear();
            c.cleanTallies();
        }

        // Assign by X's to nearest clusters (Maximization step 1)
        for (ClusterPointPair cxp : D) {
            if (this.potentialNovels.contains(cxp.x)) { // could also be if cxp.c == null, but this is safer
                continue; // ignore the outliers for a moment
            }
            final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(subclusters, cxp.x);
//            double ds[] = new double[nearestClusters.length];
//            int foo = 0;
//            for(NearestClusterTuple gnarf : nearestClusters) {
//                ds[foo++] = gnarf.getDistance();
//            }
            
            cxp.c = nearestClusters[0].getCluster();

            nearestClusters[0].getCluster().instances.add(cxp.x);
            if (cxp.x.weight() > 0.99) {
                nearestClusters[0].getCluster().addLabeling((int) cxp.x.classValue(), cxp.x.weight());
            }
        }

        // Find new radius (Expectation step)
        for (Riffle c : subclusters) {
            ret += c.recomputeAll();
        }

        // Remove empty clusters to make room for splits (Expectation-ish)
        Iterator<Riffle> cIter = subclusters.iterator();
        while (cIter.hasNext()) {
            Riffle rc = cIter.next();
            if (rc.instances.size() < 1) {
                cIter.remove();
            }
        }

        // Are we full?
        if (subclusters.size() < maxK) {
            // Fix bad clusters (Maximization step 2 - breaking up noisy clusters)
            Riffle sortedClusters[] = new Riffle[subclusters.size()];
            int tmpIdx = 0;
            for (Riffle tmpRfl : subclusters) {
                if (tmpIdx >= sortedClusters.length) {
                    break;
                }
                sortedClusters[tmpIdx] = tmpRfl;
                tmpIdx++;
            }
            Arrays.sort(sortedClusters, new Comparator<Riffle>() {
                @Override
                public int compare(Riffle first, Riffle second) {
                    if (first == null) {
                        return 1;
                    }
                    if (second == null) {
                        return -1;
                    }
                    double[] votes1 = first.getVotes().clone();
                    double[] votes2 = second.getVotes().clone();
                    double total1 = weka.core.Utils.sum(votes1);
                    double total2 = weka.core.Utils.sum(votes2);
                    Arrays.sort(votes1);
                    Arrays.sort(votes2);
                    double pentultimate1 = 1e-16 + ((votes1.length > 1) ? votes1[votes1.length - 2] : 0);
                    double pentultimate2 = 1e-16 + ((votes2.length > 1) ? votes2[votes2.length - 2] : 0);
                    // this is equiv to purity - margin... yea... really... it's awesome... gotta love math...
                    double score1 = (total1 > 0) ? first.size() * pentultimate1 / total1 : 0;
                    double score2 = (total2 > 0) ? second.size() * pentultimate2 / total2 : 0;
                    return Double.compare(score2, score1);
                }
            }
            ); // end Anon sort
            for (int cIdx = 0; cIdx < sortedClusters.length && subclusters.size() < maxK; cIdx++) {
                Riffle splitMe = sortedClusters[cIdx];
                if ( splitMe.getPurity() > 0.9) { continue;}
                double[] votes = splitMe.getVotes();
                final double totalVotes = weka.core.Utils.sum(votes);
                final double critVotes = 1.0 / (votes.length * 2);
                if (totalVotes < 2) {
                    continue;
                }
                ArrayList<Riffle> splitSet = new ArrayList<>(votes.length);
                int numberOfNewClusters = 0;
                for (int lblIdx = 0; lblIdx < votes.length; ++lblIdx) {
                    double labelVote = votes[lblIdx] / totalVotes;
                    if (labelVote >= critVotes) {
                        splitSet.add(this.createNewCluster(splitMe.toInstance()));
                        numberOfNewClusters++;
                    } else {
                        splitSet.add(null);
                    }
                }
                if (numberOfNewClusters < 2) {
                    continue;
                }
                Instances extras = new Instances(splitMe.getHeader());
                for (Instance x : splitMe.instances) {
                    if (x.weight() > 0.999) {
                        Riffle myHopefulCluster = splitSet.get((int) x.classValue());
                        if (myHopefulCluster != null) {
                            myHopefulCluster.instances.add(x);
                            myHopefulCluster.addLabeling((int) x.classValue(), x.weight());
                        } else {
                            extras.add(x);
                        }
                    } else {
                        extras.add(x);
                    }
                }
                LinkedList<Riffle> goodSet = new LinkedList<>();
                for (Riffle rfc : splitSet) {
                    if (rfc == null) {
                        continue;
                    }
                    rfc.recomputeAll();
                    goodSet.add(rfc);
                    subclusters.add(rfc);
                }
                for (Instance x : extras) {
                    final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(goodSet, x);
                    nearestClusters[0].getCluster().instances.add(x);
                }
                subclusters.remove(splitMe);
            }
        }
        
        // The pentultimate Expectation step
        ret = 0;
        for (Riffle c : subclusters) {
            ret += c.recomputeAll();
        }

        // See if any outliers should actually be consumed by a cluster now... (Maximization step 3)
        Iterator<Instance> xIter = potentialNovels.iterator();
        while (xIter.hasNext()) {
            Instance xOut = xIter.next();
            final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(subclusters, xOut);
            if (nearestClusters == null || nearestClusters.length < 1) {
                continue;
            }
            Riffle c = nearestClusters[0].getCluster();
            double d = nearestClusters[0].getDistance();
            if (d > c.getRadius()) { // Welcome home wayward tuple!
                c.instances.add(xOut);
                xIter.remove();
            }
        }

        // And the final Expectation step
        ret = 0;
        for (Riffle c : subclusters) {
            ret += c.recomputeAll();
        }
        // 
        return ret;
    } // end EM-Super-Step
    
    
    /**
     * Wrapper for parallel K-Means for processing warm-up data set
     *
     * @param D Warm-up data set
     * @param K number of clusters
     * @param useLabels if true, use
     * @return
     */
    protected final Set<Riffle> batchCluster(List<Instance> D, int K, boolean useLabels) {
        assert K >= 2 : "Minimum number of clusters (K) is 2";
        TreeSet<Riffle> ret = new TreeSet<>();
        TreeSet<Integer> labels = new TreeSet<>();
        TreeMap<Integer, TreeSet<Riffle>> potentialClusters = new TreeMap<>();
        LinkedList<ClusterPointPair> DSet = new LinkedList<>();
        //Create a potential cluster pool. Seperate into seperate pools by label if useLabels is set to true:
        for (Instance x : D) {
            int label = (useLabels) ? (int) x.classValue() : 0;
            labels.add(label);
            TreeSet<Riffle> clusterSet = potentialClusters.get(label);
            if (clusterSet == null) {
                clusterSet = new TreeSet<>();
            }
            clusterSet.add(this.createNewCluster(x));
            potentialClusters.put(label, clusterSet);
            DSet.addLast(new ClusterPointPair(x, null));
        }

        // Initialize following the K-Means++ approach:
        Riffle C = potentialClusters.firstEntry().getValue().first();
        ret.add(C);
        potentialClusters.firstEntry().getValue().remove(C);

        Iterator<Integer> labelIter = labels.iterator();
        while ((ret.size() < K) && !potentialClusters.isEmpty()) {
            if (!labelIter.hasNext()) {
                labelIter = labels.iterator();
            } // loop around as needed
            int pseudoLabel = labelIter.next();
            TreeSet<Riffle> clusterSet = potentialClusters.get(pseudoLabel);
            if (clusterSet.isEmpty()) {
                potentialClusters.remove(pseudoLabel);
                labelIter.remove();
                continue;
            }
            NearestClusterTuple[] nearestClusters = findMostLikelyClusters(clusterSet, C.toInstance());
            if (nearestClusters.length == 0) { 
                continue; 
            }
            if (nearestClusters.length == 1) {
                C = nearestClusters[0].getCluster();
            } else {                
                C = nearestClusters[nearestClusters.length - 1].getCluster();  // WAS BACKWARDS
            }
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

            EMStep(DSet, ret, this.maximumNumberOfClusterSizeOption.getValue() - (int)(this.clustersPerLabelOption.getValue() * 0.75)); // Expectation Step

            ValIdxTupleType currentScore = new ValIdxTupleType(ret);
            if (lastScore != null) {
                double diff = Math.abs(lastScore.getValIdx() - currentScore.getValIdx());
                double denominator = lastScore.getValIdx();
                valIdxDelta = (denominator == 0) ? 0.0 : Math.abs(diff / denominator);
            }
            lastScore = currentScore;
        } // end while
        return ret;
    } // end batchCluster()

    
    BufferedWriter ncCSVwriter = null; // DEBUG
    
    /**
     * Uses methodology from Kim et al. "A Novel Validity Index for Determination of the Optimal Number of Clusters"
     *
     * @param D Warm-up data set
     */
    public final void initialize(List<Instance> D) {
        String ncCSVfilePrefix = "META-" + D.get(0).dataset().relationName() + "-" + iso8601FormatString.format(new Date());
        final boolean doMetaLog = logMetaRecordsOption.isSet();
        if (doMetaLog) {
            try {
                File ncCSVFile = new File(ncCSVfilePrefix + ".csv");
                ncCSVwriter = new BufferedWriter(new FileWriter(ncCSVFile));
                String ncCSVHeader = ""
                        + "usize" + ","
                        + "urad" + ","
                        + "ctally" + ","
                        + "cpur" + ","
                        + "csize" + ","
                        + "cweight" + ","
                        + "crad" + ","
                        + "cdist" + ","
                        + "pout" + ","
                        + "vweight" + ","
                        + "qdmin" + ","
                        + "qdout" + ","
                        + "qnsc" + ","
                        + "novel";
                ncCSVwriter.write(ncCSVHeader);
                ncCSVwriter.newLine();
                ncCSVwriter.flush();
            } catch (IOException fileSetupIOException) {
                System.err.println("NC-CSV meta-data file failed to open: " + fileSetupIOException.toString());
            }
        }
        knownLabels = new int[D.get(0).numClasses()];
        Arrays.fill(knownLabels, 0);
        this.numAttributes = D.get(0).numAttributes();
        universalProbabilitySums = 0;
        bestProbabilitySums = 0;
        bestProbabilityCount = 0;
        // Setup the universal set/cluster. Note that this will be crucial for subspace selection (cross-entropy checks against null hypothesis)
        double[] universalCentroid = new double[D.get(0).numAttributes()];
        double[] universalVariance = new double[D.get(0).numAttributes()];
        Arrays.fill(universalCentroid, 0);
        Arrays.fill(universalVariance, 0);
        universalCluster = new Riffle(D.get(0));
        //universalCluster.updateStrategyOption.setChosenIndex(this.updateStrategyOption.getChosenIndex());
        //universalCluster.outlierDefinitionStrategyOption.setChosenIndex(this.outlierDefinitionStrategyOption.getChosenIndex());
        universalCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
        //universalCluster.initialStandardDeviationOption.setValue(this.initialStandardDeviationOption.getValue());
        //universalCluster.alphaAdjustmentWeightOption.setValue(this.learningRateAlphaOption.getValue());
        //universalCluster.setParentClusterer(this);
        if (D.size() > 1) {
            double[] ep = new double[universalCentroid.length];
            Arrays.fill(ep, 0);
            universalCluster.setCenter(universalCentroid); // temporary - start with standard gaussian, gets updated below
            universalCluster.setVariances(universalVariance); // temporary - start with standard gaussian, will update below
            universalCluster.setWeight(0);
            double N = D.size();
            for (Instance x : D) { // Pre-populate univeral cluster with data points
                int y = (int) x.classValue();
                if (y < knownLabels.length) {
                    knownLabels[y]++;
                }
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
                universalVariance[i] = (universalVariance[i] - ep[i] * ep[i] / N) / (N - 1);
            }
            universalCluster.setCenter(universalCentroid); // temporary - start with standard gaussian, gets updated below
            universalCluster.setVariances(universalVariance);
        }
        universalCluster.recompute(); // this updates entropies and such
        int numKnownLabels = 0;
        for (int y : knownLabels) {
            if (y > 0) {
                numKnownLabels++;
            }
        }
        // Ok, now let's use K-Means to find the initial cluster set
        int Cmin = this.clustersPerLabelOption.getValue() * numKnownLabels;
        int Cmax = Cmin + 1;
        if (optimizeInitialClusterNumberOption.isSet()) {
            Cmin = this.minimumNumberOfClusterSizeOption.getValue();//Math.max(knownLabels.size(), 2);
            Cmax = Math.max(Cmin + 1, Math.min(this.clustersPerLabelOption.getValue() * numKnownLabels, this.maximumNumberOfClusterSizeOption.getValue()));
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
        for (ValIdxTupleType i : valIdxSet) {
            i.setVo_min(CVI.getVo_min());
            i.setVo_max(CVI.getVo_max());
            i.setVu_min(CVI.getVu_min());
            i.setVu_max(CVI.getVu_max());
        }

        // Find the best K by finding the minimum score:
        valIdxSet.stream().filter((i) -> (i.getValIdx() < CVI.getValIdx())).forEach((i) -> {
            CVI = i;
        });
        
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
            for (Instance x : c.instances) {
                this.hopperCache.push(new ClusterPointPair(x, c));
            }
            this.clusters.add(c);
        }

        this.newClusterCreateCalls = 0;
        System.out.println("Starting with " + this.clusters.size() + " clusters.");
        instancesSeen = D.size();
        weightsSeen = D.size();
    } // end initialize()

    /**
     * In cases where this class is not used by the moa.tasks.EvaluateNonStationaryDynamicStream task, this safety
     * (fallback) initialization procedure is necessary.
     *
     * @param x
     */
    public final void safeInit(Instance x) {
        if (this.universalCluster == null) {
            universalCluster = new Riffle(x);
            universalCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
            double[] initialVariances = new double[x.numAttributes()];
            Arrays.fill(initialVariances, 1.0);
            universalCluster.setVariances(initialVariances);
            universalCluster.setWeight(0);
            universalCluster.recompute();
            bestProbabilitySums = 0;
            bestProbabilityCount = 0;
        }
        if (this.knownLabels == null) {
            this.knownLabels = new int[x.numClasses()];
            Arrays.fill(knownLabels, 0);
            this.numAttributes = x.numAttributes();
        }
        if (this.header == null) {
            this.header = AbstractNovelClassClassifier.augmentInstances(x.dataset());
        }
    }

    /**
     * Handle training instance that is an outlier to our current model
     * @param x Data instance
     * @param ncx nearest cluster (and distance)
     * @return cluster (if created) or null (if a total outlier)
     */
    private Riffle trainOnOutlierInstance(Instance x, NearestClusterTuple ncx) {
        Riffle ret = null;
        final boolean belowClusterLimit = (clusters.size() < this.maximumNumberOfClusterSizeOption.getValue());
        final NearestInstanceTuple[] nearestOutliers = findNearestOutliers(x);
        final int q = this.minimumClusterSizeOption.getValue();
        double qDout = 0;
        double qDmin = 0;
        if (nearestOutliers.length > q) {
            for (int i = 0; i < nearestOutliers.length && i < q; ++i) {
                qDout += nearestOutliers[i].d / (double) q;
            }
            final NearestInstanceTuple[] nearestClusterInstances = findNearestNeighbors(ncx.getCluster().instances, x);
            for (int i = 0; i < nearestClusterInstances.length && i < q; ++i) {
                qDmin += nearestClusterInstances[i].d / (double) Math.min(q, nearestOutliers.length);
            }
        }
        final double qNSC = (nearestOutliers.length >= q && (qDout > 0 || qDmin > 0)) ? (qDmin - qDout) / Math.max(qDmin, qDout) : -1.5;
        final boolean necessaryCriteriaForNewCluster = (qNSC > 0) && (nearestOutliers.length > q);

        if (necessaryCriteriaForNewCluster) { // X has a critical mass of friendly outcasts, so make a new club
            Riffle newCluster = this.createNewCluster(x);
            ret = newCluster;
            // Make new cluster up to radius of nearest cluster, but no more than 2q instances
            for (int i = 0; i < nearestOutliers.length && i < (q); ++i) {
                if (nearestOutliers[i].d > ncx.getCluster().getRadius()) {
                    break;
                }
                newCluster.addInstance(nearestOutliers[i].x);
                newCluster.instances.add(nearestOutliers[i].x);
                newCluster.trainEmbeddedClassifier(nearestOutliers[i].x);
            }

            for (Instance otherPts : ncx.getCluster().instances) {
                if (this.clustererRandom.nextDouble() < 0.5 && otherPts.weight() > 0.99) {
                    newCluster.trainEmbeddedClassifier(otherPts);
                }
            } //end for(x)
            // If at limit, prune the worst cluster to make room for this new one
            if (!belowClusterLimit) {
                double worstWeight = Double.MAX_VALUE;
                Riffle worstCluster = null;
                for (Riffle rfc : clusters) {
                    if (rfc.getWeight() < worstWeight) {
                        worstWeight = rfc.getWeight();
                        worstCluster = rfc;
                    }
                }
                if (worstCluster != null) {
                    clusters.remove(worstCluster);
                }
            }
            newCluster.recomputeAll();
            this.clusters.add(newCluster);
        }
        return ret;
    }

    /**
     * Train on data instance
     *
     * @param x instance to train on
     */
    @Override
    public final void trainOnInstanceImpl(Instance x) {
        safeInit(x);
        assert (x != null) : "Sieve::trainOnInstanceImpl() Training on a null instance!";
        int y = (int) x.classValue();
        if ((y > 0) && (y < knownLabels.length)) {
            knownLabels[y] += x.weight();
        }
        this.instancesSeen++;
        this.weightsSeen += x.weight();
        this.universalCluster.addInstance(x);
        final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(this.clusters, x);
        if (nearestClusters.length < 1 ) { // Handles weird corner case
            Riffle firstCluster = this.createNewCluster(x);
            clusters.add(firstCluster);
            System.err.println("Sieve::trainOnInstanceImpl() - no other clusters found!");
                    
        } else {
            // Everyone takes a weight hit, and we will reward the best later...
            for (NearestClusterTuple nct : nearestClusters) {
                nct.getCluster().penalize();
            }
            NearestClusterTuple ncx = nearestClusters[0]; // For code convienance
            ClusterPointPair cxp = new ClusterPointPair(x,ncx.getCluster()); // we will change this later in the function... maybe

            if (ncx.getDistance() > ncx.getCluster().getRadius() ) { // outlier
                // Hang out with the outcasts and see if you can start your own clique
                cxp.c = null;
                if (!onlyCreateNewClusterAtResyncOption.isSet()) {
                    cxp.c = trainOnOutlierInstance(x, ncx);
                }
                if (cxp.c == null) {
                    this.potentialNovels.add(x);// or just wait patiently for a friend to sit next to you
                }                
            } else { // end if(isRadialOutlier)                 
                // Or join an existing club if you are in the "IN" crowd...
                Riffle nc = ncx.getCluster();
                nc.reward();
                nc.trainEmbeddedClassifier(x);
                nc.addInstance(x);
            } // end else (not Outlier)
            // Randomly (based on distance) cross-train other models
            for (int i = 0; i < nearestClusters.length; ++i) {
                double pTrain = ((double) nearestClusters.length - i) / (2.0 * nearestClusters.length);
                if (this.clustererRandom.nextDouble() < pTrain) {
                    nearestClusters[i].getCluster().trainEmbeddedClassifier(x);
                }
            } // end for(i)
            hopperCache.addLast(cxp);
        } // corner case safety
        periodicResync();
    }
    
    /**
     * Test if we need periodic resynch yet, and if so, update via super-EM (*-Means) algorithm
     */
    private void periodicResync() {
        resyncDelayCount++;
        if (resyncDelayCount > resynchIntervalOption.getValue()) {
            // Clean hopper to correct size
            while (this.hopperCache.size() > this.cacheSizeOption.getValue()) {
                potentialNovels.remove(this.hopperCache.pop().x); // remove it from the hopper and the Outlier cache
            }
            
         if (!onlyCreateNewClusterAtResyncOption.isSet()) {
             // In this strategy, we comb through the outlier pool and extract new candidate clusters
            HashSet<Instance> removalSet = new HashSet<>();
            for(Instance fOut : this.potentialNovels) {
                if (removalSet.contains(fOut)) { 
                    continue;
                }
                final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(this.clusters, fOut);
                if (nearestClusters == null || nearestClusters.length < 1) {
                    continue;
                }
                Riffle newCluster = trainOnOutlierInstance(fOut, nearestClusters[0]);
                if (newCluster != null) {
                    removalSet.add(fOut);
                    for(Instance removeMe : newCluster.instances) {
                        removalSet.add(removeMe);
                    }
                }
            }
            for(Instance discoveredX : removalSet) {
                this.potentialNovels.remove(discoveredX);
            }
          }
            
            final double minDelta = 0.0001;
            double valIdxDelta = 1.0;
            double currentScore = 0;
            double lastScore = Double.MAX_VALUE / 10.0;
            // Iterate on E-M (really E-M-e-m-e-m)
            for (int iteration = 0; (iteration < loopsPerIterationOption.getValue()) && (valIdxDelta > minDelta); iteration++) {
                currentScore =  EMStep(hopperCache, clusters, this.maximumNumberOfClusterSizeOption.getValue() - (int)(this.clustersPerLabelOption.getValue() * 0.75));   // Maximization Step
                double diff = Math.abs(lastScore - currentScore);
                double denominator = lastScore;
                valIdxDelta = (denominator == 0) ? 0.0 : Math.abs(diff / denominator);
                lastScore = currentScore;
            } // end while
            universalCluster.instances.clear();
            universalCluster.cleanTallies();
            for (ClusterPointPair rpp : this.hopperCache) {
                universalCluster.trainEmbeddedClassifier(rpp.x);
                //universalCluster.instances.add(cxp.x);
                universalCluster.addInstance(rpp.x);
            }
            this.universalCluster.recomputeAll();
            // Prune useless clusters
            Iterator<Riffle> cIter = this.clusters.iterator();
            while(cIter.hasNext()) {
                Riffle rc = cIter.next();
                if (rc.instances.size() < 1) {
                    cIter.remove();
                }
            }
            crossTrainClusters();
            resyncDelayCount = 0;
        } // end re-sync block
    }

    /**
     * Todo - move to utils package
     * @param v
     * @param h 
     */
    public final static void safeNormalize(double[] v, Instances h) {
        int outlierIdx = h.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
        if (outlierIdx < 0) {
            outlierIdx = h.numAttributes();
        }
        double sum = 0;
        for (int i = 0; i < v.length && i < outlierIdx; ++i) {
            sum += v[i];
        }
        if (sum != 0) {
            for (int i = 0; i < v.length && i < outlierIdx; ++i) {
                v[i] /= sum;
            }
        }
    }

    /**
     * Todo - move to utils package
     * @param v
     * @param h
     * @return 
     */
    public final static int maxNonOutlier(double[] v, Instances h) {
        int outlierIdx = h.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
        if (outlierIdx < 0) {
            outlierIdx = h.numAttributes();
        }
        int maxIdx = 0;
        double maxVal = 0;
        for(int i = 0; i < v.length && i < outlierIdx ; ++i) {
            if (v[i] > maxVal) {
                maxIdx = i;
                maxVal = v[i];
            }
        }
        return maxIdx;
    }
    
    
    /**
     * Use the ensemble of NaiveBayes learners to estimate the label, assuming it is not an outlier or novel
     * @param inst Instance in question
     * @param nearestClusters order set of nearest clusters
     * @return vote array tabulating prediction
     */
    public double[] getPureLabelVotesForInstance(Instance inst, NearestClusterTuple[] nearestClusters) {
        double[] votes = new double[header.numClasses()];
        final int strategy = this.votingStrategyOption.getChosenIndex();
        double smallestDistance = 0;
        double clusterVotes[] = null;
        // Gather data
        for (NearestClusterTuple nct : nearestClusters) {
            double d = nct.getDistance(); // actually dissimilarity/distance
            clusterVotes = nct.getCluster().getVotesForInstance(inst);
            if (d == 0) {
                d = weka.core.Utils.SMALL;
            }
            smallestDistance = Math.min(d, smallestDistance);
            int localWinner = weka.core.Utils.maxIndex(nct.getCluster().getVotesForInstance(inst));
            switch (strategy) {
                case 0: // 1-NN - usually not the strongest
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] = clusterVotes[i];
                    }
                    break;
                case 1: // Global  k-NN - this is a poor performer
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i];
                    }
                    break;
                case 2: // Globally probability-weighted k-NN - good, but biased towards heavy clusters
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i] / d;
                    }
                    break;
                case 3: // Globally probability-utility-weighted k-NN - good, but overly complex
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i] * nct.getCluster().getEntropy() / d;
                    }
                    break;
                case 4: // Globally normalized k-NN - this is also usually a really really poor performer. Don't use it
                    safeNormalize(clusterVotes, header);
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i];
                    }
                    break;
                case 5: // Globally normalized probability-weighted k-NN - a safe bet
                    safeNormalize(clusterVotes, header);
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i] / d;
                    }
                    break;
                case 6: // Globally normalized probability-utility-weighted k-NN - default and preferred method
                    safeNormalize(clusterVotes, header);
                    for (int i = 0; i < clusterVotes.length; ++i) {
                        votes[i] += clusterVotes[i] * nct.getCluster().getEntropy() / d;
                    }
                    break;
                case 7: // Globally weighted k-NN winner take all per cluster - Can avoid noise, but not usually the best
                default:
                    votes[localWinner] += 1.0 / d;
            } // end switch
            if (strategy == 0) {
                break;
            }
        }
        return votes;
    }
    
    /**
     * Two conditions can fulfill the novel criteria:
     * (1) Data point is within a cluster (of sufficient size) without labels
     * (2) Data point could form its own cluster in the outlier set, and is itself an outlier from the nearest cluster
     * @param inst Instance in question
     * @param votes current vote list
     * @param nearestClusters order set of nearest clusters
     */
    public void novelClassDetection(Instance inst, double[] votes, NearestClusterTuple[] nearestClusters) {
        final int novelClassLabel = header.classAttribute().indexOfValue(AbstractNovelClassClassifier.NOVEL_LABEL_STR);
        final int outlierLabelIdx = header.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
        Riffle ncx = null;
        double ncxDist = 0;
        double rawTally = 0;
        double qDout = 0;
        double qDmin = 0;
        double qNSC = -2; // Default to very not novel
        for(NearestClusterTuple potentialNcx : nearestClusters) {// WORK AROUND - sometimes we get empty clusters???
            if(!potentialNcx.getCluster().instances.isEmpty()) {
                ncx = potentialNcx.getCluster();
                ncxDist = potentialNcx.getDistance();
                break; // stop as soon as we get a non-emtpy match
            }
        }
        if (ncx == null) { 
            votes[outlierLabelIdx] = 0.66;
            return; // really weird if this happens... let's just call it an outlier and move on...
        }
        // Test to see if we are an outlier to the nearest cluster
        if (ncxDist <= ncx.getRadius()) { // member of a cluster
            double rawLabelFrequencyCount[] = ncx.getVotes();
            rawTally = weka.core.Utils.sum(rawLabelFrequencyCount);
            if ((rawTally < 1) /*&& (ncx.size() >= this.minimumClusterSizeOption.getValue())*/) {
                votes[novelClassLabel] = 0.99; // encoding to show why novel
                votes[outlierLabelIdx] = 0; // If novel, don't call it an outlier (don't want to re-try it!)
                qNSC = 1.5; // Novel by reason of member of generated but unlabeled cluster
            }
        } else if (novelIfOutlierOption.isSet()) { // if not doing q-NSC(x) and pt is not inside nearest cluster
            votes[novelClassLabel] = 0.99; // encoding to show why novel
            votes[outlierLabelIdx] = 0; // If novel, don't call it an outlier (don't want to re-try it!)
            qNSC = 1.25; // Novel by reason of member of generated but unlabeled cluster
        } else if (this.useqNSCAtTestOption.isSet()) { // outlier
            final NearestInstanceTuple[] nearestOutliers = findNearestOutliers(inst);
            final int q = this.minimumClusterSizeOption.getValue();

            if (nearestOutliers.length > q) {
                for (int i = 0; i < nearestOutliers.length && i < q; ++i) {
                    double lqd = 0;
                    if (nearestOutliers[i] != null && Double.isFinite(nearestOutliers[i].d) && (nearestOutliers[i].d < 1.7e300)) {
                        lqd = nearestOutliers[i].d;
                    }
                    qDout += lqd / (double) q;
                    rawTally += nearestOutliers[i].x.weight();
                }
                final NearestInstanceTuple[] nearestClusterInstances = findNearestNeighbors(ncx.instances, inst);
                for (int i = 0; i < nearestClusterInstances.length && i < q; ++i) {
                    double lqd = 0;
                    if (nearestClusterInstances[i] != null && Double.isFinite(nearestClusterInstances[i].d) && (nearestClusterInstances[i].d < 1.7e300)) {
                        lqd = nearestClusterInstances[i].d;
                    }
                    qDmin += lqd / (double) Math.min(q, nearestOutliers.length);
                }
            }
            qNSC = (nearestOutliers.length >= q && (qDout > 0 || qDmin > 0)) ? (qDmin - qDout) / Math.max(qDmin, qDout) : -1.5; // not novel by reason of sparsity/non-cohesion
            if (qNSC > 0) {
                if (rawTally < 1) {
                    votes[novelClassLabel] = 0.90; // encoding to show why novel
                    votes[outlierLabelIdx] = 0; // If novel, don't call it an outlier (don't want to re-try it!)
                } else {
                    qNSC = -1.25;
                }
            }
        } else {
            votes[outlierLabelIdx] = 1;

        }
        debugMetrics( qNSC,  qDout,  qDmin,  ncxDist,  rawTally,  inst,  ncx);
    }
  
    /**
     * Find the nearest cluster, and use its most frequent label. If nearest cluster has no label, then we have a novel
     * cluster Unless data point is an outlier to all clusters, then it is just an outlier
     *
     * @param inst
     * @return
     */
    @Override
    public final double[] getVotesForInstance(Instance inst) {
        this.safeInit(inst);
        final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(this.clusters, inst);
        final double votes[] = getPureLabelVotesForInstance(inst, nearestClusters);
        if (nearestClusters.length < 1) {
            System.err.println("Sieve::getVotesForInstance() - no clusters to use for voting!");
            if (this.universalCluster != null) {
                return this.universalCluster.getVotesForInstance(inst); // Fallback plan... shouldn't happen though
            }
        }
        safeNormalize(votes, this.header);
        novelClassDetection(inst, votes, nearestClusters);
        return votes;
    }

    /**
     * Temporary function for algorithm analysis
     */
    private void debugMetrics(double qNSC, double qDout, double qDmin, double dist, double rawTally, Instance x, Riffle c) {
        if (this.logMetaRecordsOption.isSet()) {
            try {
                int groundTruth = (int) x.classValue();
                boolean isTrueNovel = (groundTruth > 0) && (groundTruth < knownLabels.length) && (knownLabels[groundTruth] < (this.minimumClusterSizeOption.getValue()));
                String ncCSVLine = ""
                        + universalCluster.size() + ","
                        + universalCluster.getRadius() + ","
                        + rawTally + ","
                        + c.getPurity() + ","
                        + c.size() + ","
                        + c.getWeight() + ","
                        + c.getRadius() + ","
                        + dist + ","
                        + (c.isOutlier(x) ? 1 : 0) + ","
                        + x.weight() + ","
                        + qDmin + ","
                        + qDout + ","
                        + qNSC + ","
                        + isTrueNovel;
                ncCSVwriter.write(ncCSVLine);
                ncCSVwriter.newLine();
                ncCSVwriter.flush();
            } catch (IOException fileIoExcption) {
                System.err.println("Could not write NC CSV line: " + fileIoExcption.toString());
            }
        }
    }


    /**
     * Wraps new cluster creation steps
     *
     * @param exemplar
     * @return newly created cluster
     */
    protected final Riffle createNewCluster(Instance exemplar) {
        Riffle newCluster = new Riffle(exemplar);
        newCluster.distanceStrategyOption.setChosenIndex(this.distanceStrategyOption.getChosenIndex());
        newCluster.embeddedLearnerOption = embeddedLearnerOption;
        newCluster.prepareEmbeddedClassifier();
        newCluster.setWeight(1);
        double[] universeVariance = universalCluster.getVariances();
        double[] initialVariance = new double[universeVariance.length];
        for (int i = 0; i < initialVariance.length; ++i) {
            initialVariance[i] = universeVariance[i] * 0.85;
        }
        newCluster.setVariances(initialVariance);
        newCluster.setParentClusterer(this);
        newCluster.recompute();
        newClusterCreateCalls++;
        return newCluster;
    }

    /**
     * Clear and re-train the embedded learners of each cluster
     */
    protected final void crossTrainClusters() {
        for (Riffle c : this.clusters) {
            c.resetEmbeddedLearning();
        }
        for (Riffle cSrc : this.clusters) {
            for (Instance x : cSrc.instances) {
                cSrc.trainEmbeddedClassifier(x); // Always train ourselves
            }
            // Selectively train other clusters (nearest clusters get more data, farthest get least - distance-wagging)
            final NearestClusterTuple[] nearestClusters = findMostLikelyClusters(this.clusters, cSrc.toInstance());
            for(int i = 0; i < nearestClusters.length; ++i) {
                double pTrain = ((double) nearestClusters.length - i) / (2.0 * nearestClusters.length);
                for (Instance x : cSrc.instances) {
                    
                    if (this.clustererRandom.nextDouble() < pTrain) {
                            nearestClusters[i].getCluster().trainEmbeddedClassifier(x);
                        }
                } //end for(x)
            } // end for(i)
        } // end for(cSrc)
    } // end function crossTrainClusters()

    ////======================================
    ////
    //// Subspace selection methods
    ////
    ////======================================
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
     * Use KL-Divergence subspace selection strategy to re-weight the attributes of each cluster As much as MatLab-style
     * code annoys me in Java/C++ code, this formula setup is complex enough to merit it here at least until it is all
     * debugged - then it can be made slightly more efficient.
     */
    protected void updateSubspaceSelectionWeightsViaKLDivergence() {
        double[] baseMeans = this.universalCluster.getCenter();
        double[] baseVariances = this.universalCluster.getVariances();

        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        for (Riffle c : this.clusters) {
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
            for (int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = (KLDistances[i] <= threshold) ? 1 : 0;
            }
            c.setAttributeWeights(newWeights);
        };
    } //end updateSubspaceSelectionWeightsViaKLDivergence()

    /**
     * Use KL-Divergence Normalized Weighing strategy to re-weight the attributes of each cluster As much as MatLab-style
     * code annoys me in Java/C++ code, this formula setup is complex enough to merit it here at least until it is all
     * debugged - then it can be made slightly more efficient.
     */
    protected void updateSubspaceSelectionWeightsViaKLNorm() {
        double[] baseMeans = this.universalCluster.getCenter();
        double[] baseVariances = this.universalCluster.getVariances();

        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        for (Riffle c : this.clusters) {
            double[] clusterVariances = c.getVariances();
            double[] clusterMeans = c.getCenter();
            double[] KLDistances = new double[baseVariances.length];
            // Compute the K-L Divergance metric for each attribute independantly
            for (int i = 0; i < KLDistances.length && i < clusterMeans.length; ++i) {
                double DMean = baseMeans[i];
                double DVar = Math.max(baseVariances[i], weka.core.Utils.SMALL);
                double cMean = clusterMeans[i];
                double cVar = Math.max(clusterVariances[i], weka.core.Utils.SMALL);
                KLDistances[i] = Math.max(VectorDistances.KLDiverganceGaussian(cMean, cVar, DMean, DVar), weka.core.Utils.SMALL);
            } // end for(attributes)

            weka.core.Utils.normalize(KLDistances);
            c.setAttributeWeights(KLDistances);
        };
    } // end updateSubspaceSelectionWeightsViaKLNorm()

    /**
     * Use Variance subspace selection strategy to re-weight the attributes of each cluster Do not expect this option to
     * gain much, but it is here for comparative testing.
     */
    protected void updateSubspaceSelectionWeightsViaNormalizedVariance() {
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        for (Riffle c : this.clusters) {
            double[] clusterVariances = c.getVariances();

            double[] newWeights = new double[clusterVariances.length];
            for (int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = Math.min(clusterVariances[i], weka.core.Utils.SMALL);
            }
            weka.core.Utils.normalize(newWeights);
            c.setAttributeWeights(newWeights);
        };
    }

    /**
     * Use Gain-like construct for subspace selection strategy to re-weight the attributes of each cluster
     */
    protected void updateSubspaceSelectionWeightsViaGain() {
        double[] baseEntropies = this.universalCluster.getEntropies();
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        for (Riffle c : this.clusters) {
            double[] clusterEntropies = c.getEntropies();
            double[] newWeights = new double[clusterEntropies.length];
            // Compute the K-L Divergance metric for each attribute independantly
            double EntropyTotal = 0;
            for (int i = 0; i < baseEntropies.length && i < clusterEntropies.length; ++i) {
                double Hu = baseEntropies[i];
                double Hc = clusterEntropies[i];
                double diff = (Hu == 0) ? 0 : (Hu - Hc) / Hu;
                if (Double.isNaN(diff)) {
                    diff = weka.core.Utils.SMALL;
                }
                EntropyTotal += diff;
                newWeights[i] = diff;
            } // end for(attributes)
            if (EntropyTotal == 0) {
                EntropyTotal = 1;
            }
            for (int i = 0; i < newWeights.length; ++i) {
                newWeights[i] /= EntropyTotal;
            }
            c.setAttributeWeights(newWeights);
        }
    }

    /**
     * Use Variance subspace selection strategy to re-weight the attributes of each cluster Do not expect this option to
     * gain anything, but it is here for comparative testing (and sanity checks).
     */
    protected void updateSubspaceSelectionWeightsViaRandom() {
        int numAttribs = this.universalCluster.getCenter().length;
        Random rng = new Random(numAttribs);
        //Each cluster is modified independantly of each other, and uses the base "universal" cluster Read-Only,
        // so take advantage of the Java 8 parallelism opportunity...
        for (Riffle c : this.clusters) {

            double[] newWeights = new double[numAttribs];
            for (int i = 0; i < newWeights.length; ++i) {
                newWeights[i] = rng.nextDouble();
            }
            weka.core.Utils.normalize(newWeights);
            c.setAttributeWeights(newWeights);
        };
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        int numKnownLabels = 0;
        if (knownLabels != null) {
            for (int y : knownLabels) {
                if (y > 0) {
                    numKnownLabels++;
                }
            }
        }
        Measurement[] ret = new Measurement[]{
            new Measurement("NumClusters",
            this.clusters.size()),
            new Measurement("AvgClusterSize",
            this.getAverageSize()),
            new Measurement("AvgClusterLabels",
            this.getAverageLabels()),
            new Measurement("AvgClusterRadius",
            this.getAverageVariance()),
            new Measurement("UniverseRadius",
            (this.universalCluster != null) ? this.universalCluster.getRadius() : 0),
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
            numKnownLabels),
            new Measurement("NewLabelCounter",
            this.newLabelCount), // TODO - there are other measurements we probably want...
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
        if (this.clusters.isEmpty()) {
            return 1;
        }
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
        if (this.clusters.isEmpty()) {
            return 1;
        }
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
        if (this.clusters.isEmpty()) {
            return 1;
        }
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
        if (this.clusters.isEmpty()) {
            return 1;
        }
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
        if (this.clusters.isEmpty()) {
            return 1;
        }
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
        if (this.clusters.isEmpty()) {
            return 0;
        }
        int ret = 0;
        ret = clusters.parallelStream().map((c) -> c.size()).reduce(ret, (accumulator, _item) -> accumulator + _item);
        return ((double) ret) / ((double) this.clusters.size());
    }

    /**
     *
     * @return
     */
    protected double getAverageLabels() {
        if (this.clusters.isEmpty()) {
            return 0;
        }
        int ret = 0;
        ret = clusters.parallelStream().map((c) -> c.getNumLabeledPoints()).reduce(ret, (accumulator, _item) -> accumulator + _item);
        return ((double) ret) / ((double) this.clusters.size());
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        StringUtils.appendIndented(out, indent, "'Find Entities in SubSpace (FeS2) using " + this.clusters.size() + " clusters.");
        StringUtils.appendNewline(out);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public Clustering getClusteringResult() {
        this.clustering = new Clustering();
        for (Riffle c : this.clusters) {
            clustering.add(c);
        };
        return this.clustering;
    }

    /**
     * The SizeOfAgent method returns a value or -1 many times, so this override assures at least some estimate using
     * intrinsic knowledge of the object structure.
     *
     * @return Estimated numbed of bytes used by this object for data
     */
    @Override
    public int measureByteSize() {
        int ret = super.measureByteSize();
        if (ret <= 0) {
            ret = this.clusters.stream().map((c) -> c.measureByteSize()).reduce(ret, Integer::sum);
            if (this.universalCluster != null) {
                ret += this.universalCluster.measureByteSize();
            }
            if (this.hopperCache != null) {
                ret += this.hopperCache.size() * 84;
            }
            ret += 84;
        }
        return ret;
    }

    ////========================================================
    ////
    ////  PERCEPTRON METHODS FOR OUTLIER DETECTION
    ////
    //// Note: We should just use M3 with correct options instead of making a hard-coded ensemble
    ////       We would still need to create our own special data instances though with custom features
    ////
    //// Note: In the end, this option proves not as reliabel for NCD as a simple radius test
    ////========================================================
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
        attribs.add(new Attribute("isOutlier", labels)); // 4
        for (Attribute a : attribs) {
            a.setWeight(1.0);
        }
        attribs.get(attribs.size() - 1).setWeight(0);
        outlierPerceptronTrainingSet = new Instances("PerceptronTrainer", attribs, 5000 * this.clusters.size());
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
        double r = (c.getRadius() != 0) ? c.getRadius() : 1;
        //double w = c.getWeight();
        double N = Math.min(c.size(), this.cacheSizeOption.getValue());
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
        final int numberOfPerceptrons = 1;
        final int MEMBER = 0;
        final int OUTLIER = 1;
        double accuracySum = 0;
        double accuracyCount = 0;
        this.outlierPerceptronTrainingSet.clear();
        Random rng = new Random(this.randomSeed);

        // Generate training set
        for (Riffle thisCluster : this.clusters) {
            for (Instance x : thisCluster.getHeader()) {
                Instance pseudoPt = makePerceptronInstance(thisCluster, x);
                for (Riffle thatCluster : this.clusters) {
                    double groundTruth = (thisCluster == thatCluster) ? MEMBER : OUTLIER;
                    pseudoPt.setClassValue(groundTruth);
                    this.outlierPerceptronTrainingSet.add(pseudoPt);
                }
            }
        }
        for (Instance x : this.outlierPerceptronTrainingSet) {
            x.setWeight(1.0 / this.outlierPerceptronTrainingSet.numInstances());
        };

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
                    if (Double.isNaN(newWeight)) {
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
        //int outlierIdx = this.header.classAttribute().indexOfValue(AbstractNovelClassClassifier.OUTLIER_LABEL_STR);
        for (int i = 0; i < this.perceptrons.length && i < this.pweights.length; ++i) {
            double localVotes[] = this.perceptrons[i].getVotesForInstance(x);
            for (int v = 0; v < localVotes.length && v < votes.length; ++v) {
                double delta = this.pweights[i] * localVotes[v];
                votes[v] += (Double.isNaN(delta)) ? 0 : delta;
                voteSum += (Double.isNaN(delta)) ? 0 : delta;
            }
        }
        if (voteSum != 0) {
            for (int i = 0; i < votes.length; ++i) {
                votes[i] /= voteSum;
            }
        }
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
        return (1 == getPerceptronVotesForOutlierStatus(pseudoInstance));
    }

}

