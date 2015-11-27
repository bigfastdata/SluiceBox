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

package moa.core.tuples;

import weka.core.Instance;



/**
 * Tuple to track x nearness
 * @author bparker
 */
public final class NearestInstanceTuple implements Comparable<NearestInstanceTuple> {
        public final Instance x;
        public final double d ;
        //private boolean isOutlier;
        
        public NearestInstanceTuple(Instance inst, double dist) {
            x = inst;
            d = dist;
        }
        public Instance getCluster() { return x; }
        public double getDistance() { return d;}
        
    /**
     * Used to avoid re-computing p(x) ~ N(mu, sigma) too often
     * @param other instance to compare
     * @return true if probability passes outlier test for x
     */
            
        @Override
        final public int  compareTo(NearestInstanceTuple other) {
            int ret = 0;
            if (this.d > other.d) { // Then highest probability
                ret = 1;
            } else if (this.d < other.d) {
                ret = -1;
            } else if (this.x.weight() < other.x.weight()) { // then newest (biggest) Id
                ret = 1;
            } else if (this.x.weight() > other.x.weight()) {
                ret = -1;
            }
            return ret;
            }
    }