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

import java.io.File;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import moa.classifiers.Classifier;
import moa.classifiers.novelClass.AbstractNovelClassClassifier;
import moa.clusterer.FeS2;
import moa.core.InstancesHeader;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.core.VectorDistances;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.UnsafeUtils;

/**
 * SluiceBoxRush.java
 *
 * This class was originally designed for use by the RandomMixedNovelDriftGenerator for MOA as part of Brandon Parker's
 * Dissertation work. This class is actually a thin wrapper around the FeS2 class, providing a classifier instantiation
 * of the FeS2 Clusterer. This was a failed gold-rush attempt that only found "fools gold."
 *
 * Copyright (C) 2013 University of Texas at Dallas
 *
 * @author Brandon S. Parker (brandon.parker@utdallas.edu)
 * @version $Revision: 1 $
 */
public class SluiceBoxRush extends AbstractNovelClassClassifier {

    private static final long serialVersionUID = 1L;
    protected FeS2 dynamicStreamClustering = new FeS2();
    
    public IntOption warmupLengthOption = new IntOption("warmupLength", 'W',
            "Number of data points at beginning of stream for a priori statistics",
            5000, 1, Integer.MAX_VALUE);
     
    // Pass through settings
    public final FloatOption learningRateAlphaOption = dynamicStreamClustering.learningRateAlphaOption;
    
    public final IntOption minimumClusterSizeOption = dynamicStreamClustering.minimumClusterSizeOption;
    
     public final IntOption minimumNumberOfClusterSizeOption = dynamicStreamClustering.minimumNumberOfClusterSizeOption;
    
    public final IntOption maximumNumberOfClusterSizeOption = dynamicStreamClustering.maximumNumberOfClusterSizeOption;    
    
    public final IntOption clustersPerLabelOption = dynamicStreamClustering.clustersPerLabelOption;

   // public final FloatOption pruneThresholdOption = dynamicStreamClustering.pruneThresholdOption;
    
    public final FloatOption initialClusterWeightOption = dynamicStreamClustering.initialClusterWeightOption;
    
    public final MultiChoiceOption updateStrategyOption =  dynamicStreamClustering.updateStrategyOption;
    
    public final MultiChoiceOption outlierDefinitionStrategyOption = dynamicStreamClustering.outlierDefinitionStrategyOption;
    
    //public final MultiChoiceOption distanceNormStrategyOption = dynamicStreamClustering.distanceNormStrategyOption;
    
    public final MultiChoiceOption subspaceStrategyOption = dynamicStreamClustering.subspaceStrategyOption;
    
    public final MultiChoiceOption votingStrategyOption = dynamicStreamClustering.votingStrategyOption;
    
    public final MultiChoiceOption distanceStrategyOption = dynamicStreamClustering.distanceStrategyOption;
    
    public final MultiChoiceOption positiveClusterFeedbackStrategyOption = dynamicStreamClustering.positiveClusterFeedbackStrategyOption;

    public final FlagOption optimizeInitialClusterNumberOption = dynamicStreamClustering.optimizeInitialClusterNumberOption;
    
    public final MultiChoiceOption inclusionProbabilityStrategyOption = dynamicStreamClustering.inclusionProbabilityStrategyOption;
    
    public final FloatOption hypothesisWeightOption = dynamicStreamClustering.hypothesisWeightOption;
    
//    public final FloatOption initialSigmaMulitplierOption = new FloatOption("initialSigmaMultiplier", 'x',
//            "Multiplier for initial sigma found via k-Dist curve analysis",
//            1.0, 0.001, 10000.0);
    
    protected List<Instance> warmupCache = new LinkedList<>();
    
    @Override
    public void prepareForUseImpl(TaskMonitor mon, ObjectRepository repo) {
        this.dynamicStreamClustering.resetLearning();
        System.out.println(this.getPurposeString());
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.warmupCache != null) {
          if (this.warmupCache.size() < this.warmupLengthOption.getValue())  {
              this.warmupCache.add(inst);
          } else {
              this.dynamicStreamClustering.initialize(warmupCache);
              //processWarmupCache();
              this.warmupCache.clear();
              this.warmupCache = null;
          }
        } else {
            this.dynamicStreamClustering.trainOnInstance(inst);
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.dynamicStreamClustering.getVotesForInstance(inst);
    }

    /**
     * @depricated
     */
    protected void processWarmupCache() {
        this.setModelContext(new InstancesHeader(this.warmupCache.get(0).dataset())); 
        for(int i = 0; i < this.getModelContext().numAttributes(); ++i) {
            Attribute a = this.getModelContext().attribute(i);
            double w = (this.getModelContext().classIndex() == i) ? 0.0 : 1.0;
            a.setWeight(w);
            UnsafeUtils.setAttributeRange(a, 0.0, 1.0);
        }
        int size = this.warmupCache.size();
        int k = this.minimumClusterSizeOption.getValue();
        assert(size > 3) : "Warmup size is too small";
        ArrayList<Double> distancesRow = new ArrayList<>(size);
        double kthDistances[] = new double[size];
        for(int i = 0; i < size; ++i) {
            distancesRow.clear();
            for(int j = 0; j < size; ++j) {
                Double dist = VectorDistances.distance(warmupCache.get(i).toDoubleArray(), warmupCache.get(j).toDoubleArray(), modelContext, this.distanceStrategyOption.getChosenIndex());
                if (!dist.isInfinite() && !dist.isNaN() && (dist > 0)) {
                    distancesRow.add(dist);
                }
            }
            Collections.sort(distancesRow);
            if (distancesRow.isEmpty()) {
                kthDistances[i] = 0;
            } else if (distancesRow.size() >= k) {
                kthDistances[i] = distancesRow.get(k); 
            } else {
                kthDistances[i] = distancesRow.get(distancesRow.size() - 1);
            }
        }
        Arrays.parallelSort(kthDistances);
        double idealSigma = 0.1;
        double x1 = 0.0;
        double y1 = kthDistances[0];
        double xn = kthDistances.length;
        double yn =  kthDistances[kthDistances.length - 1];
        double m = (yn - y1) / (xn - x1);
        double b = yn - (m * xn);
        double maxDistanceToLine = 0.0;
        double currentDistanceToLine = 0.0;
        for(int i = 0; i < kthDistances.length; ++i) {
            currentDistanceToLine = Math.abs((m * i + b) - kthDistances[i]);
            if (Double.isFinite(currentDistanceToLine) && currentDistanceToLine >= maxDistanceToLine) {
                maxDistanceToLine = currentDistanceToLine;
                idealSigma = kthDistances[i];
            }
        }
        if (Double.isNaN(idealSigma)) { idealSigma = 0.001;}
        idealSigma = Math.max(idealSigma, weka.core.Utils.SMALL);
        this.dynamicStreamClustering.initialStandardDeviationOption.setValue(idealSigma);
        System.out.println("Default Sigma = " + idealSigma);
        for (Instance x : this.warmupCache) {
            this.dynamicStreamClustering.trainOnInstance(x);
        }
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
        StringUtils.appendIndented(out, indent, "SluiceBox [build:" + compileDateStr+ "] Used FeS2 clustering for classification and novel class detection. ");
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
        String ret = "Fools Gold - Mine non-stationary data streams predicting data point labels for B. Parker's Dissertation (build:" + compileDateStr + ")";
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
