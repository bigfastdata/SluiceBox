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

import moa.cluster.Riffle;

/**
 * Tuple to track cluster nearness
 * @author bparker
 */
public final class NearestClusterTuple implements Comparable<NearestClusterTuple> {
        private final Riffle cluster;
        private double distance ;
        //private boolean isOutlier;
        
        public NearestClusterTuple(Riffle c, double p) {
            cluster = c;
            distance = p;
        }
        public Riffle getCluster() { return cluster; }
        public double getDistance() { return distance;}
        
        public void setDistance(double distArg) {
            distance = distArg;
        }
        
        public void multiply(double factor) {
            distance *= factor;
        }
        
    /**
     * Used to avoid re-computing p(x) ~ N(mu, sigma) too often
     *
     * @param other
     * @return true if probability passes outlier test for cluster
     */
    @Override
    final public int compareTo(NearestClusterTuple other) {
        int ret = 0;
        if (this.distance > other.distance) { // Then highest probability
            ret = 1;
        } else if (this.distance < other.distance) {
            ret = -1;
        } else if (this.cluster.getId() < other.cluster.getId()) { // then newest (biggest) Id
            ret = 1;
        } else if (this.cluster.getId() > other.cluster.getId()) {
            ret = -1;
        }
        return ret;
    }
}
