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

import java.util.Collection;
import moa.cluster.Riffle;

/**
 *
 * @author bparker
 */
public final class ValIdxTupleType {

    /**
     * Ideally we want this number very very small
     */
    protected double MICD = 1;
    protected double vu_min = 0;
    protected double vu_max = 1;
    /**
     * Ideally we want this number quite high:
     */
    protected double ICMD = 0;
    protected double vo_min = 0;
    protected double vo_max = 1;
    protected int K = 0;
    protected Collection<Riffle> clusterSet;

    public ValIdxTupleType(Collection<Riffle> V) {
        MICD = V.parallelStream().map((c) -> c.getRadius() / ((double) V.size())).reduce(MICD, (accumulator, _item) -> accumulator + _item);
        ICMD = V.parallelStream().map((c1) -> {
            double dmin = Double.MAX_VALUE;
            for (Riffle c2 : V) {
                if (c1 == c2) { continue; }
                double dcurr = c1.getCenterDistance(c2);
                if (Double.isFinite(dcurr) && dcurr < dmin) {
                    dmin = dcurr;
                }
            }
            if (Double.isNaN(dmin) || dmin <= 0) { dmin = weka.core.Utils.SMALL; }
            return dmin;
        }).map((dmin) -> ((double) V.size()) / dmin).reduce(ICMD, (accumulator, _item) -> accumulator + _item);
        this.K = V.size();
        this.clusterSet = V;
    }

    public int getK() {
        return K;
    }

    public Collection<Riffle> getClustering() {
        return clusterSet;
    }

    public void clearClustering() {
        this.clusterSet.clear();
    }
    
    public double getVo() {
        return (this.ICMD - vo_min) / (vo_max - vo_min);
    }

    public double getICMD() {
        return getVo();
    }

    public double getVu() {
        return (this.MICD - vu_min) / (vu_max - vu_min);
    }

    public double getMICD() {
        return getVu();
    }

    public double getValIdx() {
        return MICD + ICMD;
    }

    public double getVu_min() {
        return vu_min;
    }

    public double getVo_min() {
        return vo_min;
    }

    public double getVu_max() {
        return vu_max;
    }

    public double getVo_max() {
        return vo_max;
    }

    public void setVu_min(double m) {
        vu_min = m;
    }

    public void setVo_min(double m) {
        vo_min = m;
    }

    public void setVu_max(double m) {
        vu_max = m;
    }

    public void setVo_max(double m) {
        vo_max = m;
    }

    public void clear() {
        ICMD = 0;
        MICD = 0;
        vu_min = 0;
        vu_max = 1;
        vo_min = 0;
        vo_max = 1;
    }
}
