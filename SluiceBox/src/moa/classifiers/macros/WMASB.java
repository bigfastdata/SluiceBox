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

package moa.classifiers.macros;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.meta.WeightedMajorityAlgorithm;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.options.ListOption;
import moa.options.Option;
import moa.tasks.TaskMonitor;
import weka.core.Instance;

/**
 *
 * @author bparker
 */
public class WMASB  extends AbstractClassifier {
    private static final long serialVersionUID = 1L;
    private final static WeightedMajorityAlgorithm wma = new WeightedMajorityAlgorithm();
    
    public final static ListOption learnerListOptionOverride = new ListOption(
            "learners",
            'l',
            "The learners to combine.",
            new ClassOption("learner", ' ', "", Classifier.class,
            "trees.HoeffdingTree"),
            new Option[]{
                new ClassOption("", ' ', "", Classifier.class,
                "bayes.NaiveBayes"),
                new ClassOption("", ' ', "", Classifier.class,
                "functions.Perceptron -r 2.0"),
                new ClassOption("", ' ', "", Classifier.class,
                "functions.Perceptron -r 1.0"),
                new ClassOption("", ' ', "", Classifier.class,
                "functions.Perceptron -r 0.50"),
                new ClassOption("", ' ', "", Classifier.class,
                "functions.Perceptron -r 0.25"),
                },
            ',');
    
    @Override
    public String getPurposeString() {
        return "'Macro' wrapper for Weighted Maj. Algo using Naive Bayes and 4 Perceptron base learners, created for B. Parker's Dissertation to expedite batch testing";
    }
    
    @Override
    public void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        wma.learnerListOption = learnerListOptionOverride;
        wma.prepareForUseImpl(monitor, repository);
    }
    
    @Override
    public void resetLearningImpl() {
         wma.resetLearningImpl();
     }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        wma.trainOnInstanceImpl(inst);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }
    
    @Override
    public Measurement[] getModelMeasurements() {
        return wma.getModelMeasurements();
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        wma.getModelDescription(out, indent);
    }

    @Override
    public boolean isRandomizable() {
        return wma.isRandomizable();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return wma.getVotesForInstance(inst);
    }

}
