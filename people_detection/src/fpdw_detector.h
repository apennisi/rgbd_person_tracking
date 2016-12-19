/*
 *  The FASTEST PEDESTRIAN DETECTOR IN THE WEST (FPDW)
 *  Copyright 2015 Andrea Pennisi
 *
 *  This file is part of AT and it is distributed under the terms of the
 *  GNU Lesser General Public License (Lesser GPL)
 *
 *
 *
 *  FPDW is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  FPDW is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with FPDW.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  FPDW has been written by Andrea Pennisi
 *
 *  Please, report suggestions/comments/bugs to
 *  andrea.pennisi@gmail.com
 *
 */


#ifndef FPDW_DETECTOR_H
#define FPDW_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <math.h>
#include <omp.h>

#include "classifier.h"
#include "cutils.h"
#include "nms.h"

namespace fpdw
{
    namespace detector
    {
        class FPDWDetector
        {
            public:
                FPDWDetector(const std::string &_xml_file,  const float &_conf);
                FPDWDetector(const Classifier &_cls);
                void process(const cv::Mat &_img);
                const std::vector<cv::Rect> getBBoxes() const
                {
                    return m_bboxes;
                }
            private:
                Classifier m_cls;
                structs::Cls m_clf;
                cv::Mat m_img;
                std::vector<int> m_modelDsPad;
                std::vector<int> m_modelDs;
                uint m_stride;
                float m_cascThr;
                utils::GradientMag gradient;
                int m_nPerOct;
                int m_nOctUp;
                cv::Size m_minDs;
                int m_shrink;
                float m_shrink_inv;
                cv::Size m_img_size;
                std::vector<float> m_shift;
                std::vector<float> m_scales;
                std::vector<cv::Point2f> m_scaleHw;
                std::vector< std::vector<cv::Mat> > m_data;
                std::vector<int> isR, isA, isN;
                std::vector<structs::PadWith> m_padWith;
                int nScales;
                int nTypes;
                bool init_detector;
                float m_confidence;
                std::vector<cv::Rect> m_bboxes;
                //CLASSIFIER PARAMETERS
                int modelHt;
                int modelWd;
                int stride;
                float cascThr;
                float *thrs;
                float *hs;
                int *fids;
                int *child;
                int treeDepth;
                int nTreeNodes;
                int nTrees;
                int stride_shrink;
                int endWd;
                int endHt;
                Nms nms;
            private:
                void chnsPyramid(const fpdw::structs::Pyramid &_pyr);
                void chnsCompute(const cv::Mat &_img, const fpdw::structs::Chns &_chns, const int &_isr);
                inline void getChild( float *chns1, uint *cids, int *fids, float *thrs, int offset, int &k0, int &k );
                void getScale();
                void init();
                cv::Mat detect(const int &_i);
                cv::Mat vectorToCvMat(const std::vector<cv::Mat> &_v, int &nChannels);
        };
    }

}

#endif
