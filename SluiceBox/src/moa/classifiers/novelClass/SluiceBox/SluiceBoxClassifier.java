/*
 *    SluiceBox.java
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
package moa.classifiers.novelClass.SluiceBox;

import moa.clusterer.outliers.Sieve;
import java.io.File;
import java.net.URISyntaxException;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import moa.classifiers.Classifier;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.options.ClassOption;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;


/**
 * SluiceBox.java
 *
 * This class was originally designed for use by the RandomMixedNovelDriftGenerator for MOA as part of Brandon Parker's
 * Dissertation work. This class is actually a thin wrapper around the FeS2 class, providing a classifier instantiation
 * of the FeS2 Clusterer.
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 4 $
 */
public class SluiceBoxClassifier extends AbstractNovelClassClassifier {

    private static final long serialVersionUID = 1L;
    protected Sieve dynamicStreamClustering = new Sieve();
  
  //  public final FlagOption usePYForEMOption = dynamicStreamClustering.usePYForEMOption;
    public final FlagOption novelIfOutlierOption = dynamicStreamClustering.novelIfOutlierOption;
  //  public final FlagOption novelIfTooFewVotesOption = dynamicStreamClustering.novelIfTooFewVotesOption;
  //  public final FlagOption novelIfTooSmallMaxVoteOption = dynamicStreamClustering.novelIfTooSmallMaxVoteOption;
    
    public IntOption warmupLengthOption = new IntOption("warmupLength", 'W',
            "Number of data points at beginning of stream for a priori statistics. Ideally SB.Warmup + SB.TrainingDelay = ENSDS.Warmup",
            3000, 1, Integer.MAX_VALUE);
     
    // Pass through settings
    //public final FloatOption learningRateAlphaOption = dynamicStreamClustering.learningRateAlphaOption;
    public final FlagOption onlyCreateNewClusterAtResyncOption = dynamicStreamClustering.onlyCreateNewClusterAtResyncOption;
    public final IntOption resynchIntervalOption  = dynamicStreamClustering.resynchIntervalOption;
    public final IntOption loopsPerIterationOption = dynamicStreamClustering.loopsPerIterationOption;
    
    public final IntOption cacheSizeOption = dynamicStreamClustering.cacheSizeOption;
    
    public final IntOption minimumClusterSizeOption = dynamicStreamClustering.minimumClusterSizeOption;
    public final IntOption minimumNumberOfClusterSizeOption = dynamicStreamClustering.minimumNumberOfClusterSizeOption;
    public final IntOption maximumNumberOfClusterSizeOption = dynamicStreamClustering.maximumNumberOfClusterSizeOption;    
    public final IntOption clustersPerLabelOption = dynamicStreamClustering.clustersPerLabelOption;
    
   // public final FloatOption pruneThresholdOption = dynamicStreamClustering.pruneThresholdOption;
    
    //public final FloatOption initialClusterWeightOption = dynamicStreamClustering.initialClusterWeightOption;
    
    //public final MultiChoiceOption updateStrategyOption =  dynamicStreamClustering.updateStrategyOption;
    
    public final MultiChoiceOption outlierDefinitionStrategyOption = dynamicStreamClustering.outlierDefinitionStrategyOption;
    
    public final ClassOption embeddedLearnerOption = dynamicStreamClustering.embeddedLearnerOption;
    
    //public final MultiChoiceOption distanceNormStrategyOption = dynamicStreamClustering.distanceNormStrategyOption;
    
    public final MultiChoiceOption subspaceStrategyOption = dynamicStreamClustering.subspaceStrategyOption;
    
    public final MultiChoiceOption votingStrategyOption = dynamicStreamClustering.votingStrategyOption;
    
    public final MultiChoiceOption distanceStrategyOption = dynamicStreamClustering.distanceStrategyOption;
    
    public final MultiChoiceOption positiveClusterFeedbackStrategyOption = dynamicStreamClustering.positiveClusterFeedbackStrategyOption;

    public final FlagOption optimizeInitialClusterNumberOption = dynamicStreamClustering.optimizeInitialClusterNumberOption;
    
    public final MultiChoiceOption updateStrategyOption = dynamicStreamClustering.updateStrategyOption;
    
    public final MultiChoiceOption inclusionProbabilityStrategyOption = dynamicStreamClustering.inclusionProbabilityStrategyOption;
    
    public final FloatOption hypothesisWeightOption = new FloatOption("hypothesisWeight", 'h',
            "For unlabeled data, contribute the cluster labelling by this amount using h(x). Note this is dangerous as h(x) drifts from g(x)",
            0.001, 0.0, 0.999);
    
    public ClassOption sslLearnerOption = new ClassOption("SSLLearner", 'L',
            "Classifier to train for predicting label for SSL hypothesis.", Classifier.class, "meta.M3");
    
    public final FloatOption sslVoteWeightOption = new FloatOption("sslVoteWeight", 'x',
            "Relative Weighting of the SSL discriminative classifer in ration to the generative classifier",
            0.001, 0.0, 10.0);
    public final FlagOption useqNSCAtTestOption = dynamicStreamClustering.useqNSCAtTestOption;
    public final FlagOption logMetaRecordsOption = dynamicStreamClustering.logMetaRecordsOption;
    
//    public final FloatOption initialSigmaMulitplierOption = new FloatOption("initialSigmaMultiplier", 'x',
//            "Multiplier for initial sigma found via k-Dist curve analysis",
//            1.0, 0.001, 10000.0);
    
    protected List<Instance> warmupCache = new LinkedList<>();
    protected Classifier roughClassifier = null;
    
    @Override
    public void prepareForUseImpl(TaskMonitor mon, ObjectRepository repo) {
        this.dynamicStreamClustering.resetLearning();
        roughClassifier = (Classifier) getPreparedClassOption(this.sslLearnerOption);
        roughClassifier.resetLearning();
        System.out.println(this.getPurposeString());
        //novelIfOutlierOption.set();
        //novelIfTooFewVotesOption.set();
        //novelIfTooSmallMaxVoteOption.set();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        Instance pseudoPoint = augmentInstance(inst);
        
        if (inst.weight() < weka.core.Utils.SMALL) {
           double[] votes = roughClassifier.getVotesForInstance(inst);
           pseudoPoint.setClassValue(weka.core.Utils.maxIndex(votes));
           pseudoPoint.setWeight(hypothesisWeightOption.getValue());
        } else {
            roughClassifier.trainOnInstance(inst);
        }
        
        if (this.warmupCache != null) {
          if (this.warmupCache.size() < this.warmupLengthOption.getValue())  {
              this.warmupCache.add(pseudoPoint);
          } else {
              this.dynamicStreamClustering.initialize(warmupCache);
              this.warmupCache.clear();
              this.warmupCache = null;
          }
        } else {
            this.dynamicStreamClustering.trainOnInstance(pseudoPoint);
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] ret = this.dynamicStreamClustering.getVotesForInstance(inst);
        double sslVotes[] = this.roughClassifier.getVotesForInstance(inst);
        Sieve.safeNormalize(ret, inst.dataset());
        for (int i = 0; i < ret.length && i < sslVotes.length; ++i) {
            ret[i] += sslVotes[i] * sslVoteWeightOption.getValue();
        }
        return ret;
    }


    @Override
    public void resetLearningImpl() {
        this.dynamicStreamClustering.resetLearning();
        warmupCache = new LinkedList<>();
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        String compileDateStr = "(unknown)";
        try {
            File jarFile = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().toURI());
            Date compileDate = new Date(jarFile.lastModified());
            compileDateStr = compileDate.toString();
        } catch (URISyntaxException e) { }
        StringUtils.appendIndented(out, indent, "SluiceBox [build:" + compileDateStr+ "] Used Sieve clustering for classification and novel class detection. ");
        StringUtils.appendNewline(out);
        this.dynamicStreamClustering.getModelDescription(out, indent);
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    /**
     * Sub "classifiers" are really sub-clusters. Type miss-match if we tried to pass them.
     */
    public Classifier[] getSubClassifiers() {
        return null;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return this.dynamicStreamClustering.getModelMeasurements();
    }
    
    /**
     * 
     * @return purpose description of the class
     */
    @Override
    public String getPurposeString() {
        String compileDateStr = "(unknown)";
        try {
            File jarFile = new File(this.getClass().getProtectionDomain().getCodeSource().getLocation().toURI());
            Date compileDate = new Date(jarFile.lastModified());
            compileDateStr = compileDate.toString();
        } catch (URISyntaxException e) { }
        String ret = "Mine non-stationary data streams predicting data point labels for B. Parker's Dissertation (build:" + compileDateStr + ")";
        return ret;
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
            if (this.novelLabelIndex >= 0) { ret += this.novelLabelIndex * 8;}
            if (this.warmupCache != null) { ret += this.warmupCache.size() * 8;}
            if (this.dynamicStreamClustering != null) { ret += this.dynamicStreamClustering.measureByteSize();}
        }
        return ret;
    }
 }
