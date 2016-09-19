/*
 *  RGBD Persom Tracker
 *  Copyright 2016 Andrea Pennisi
 *
 *  This file is part of AT and it is distributed under the terms of the
 *  GNU Lesser General Public License (Lesser GPL)
 *
 *
 *
 *  AT is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  AT is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with AT.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  AT has been written by Andrea Pennisi
 *
 *  Please, report suggestions/comments/bugs to
 *  andrea.pennisi@gmail.com
 *
 */


#ifndef PATCH_DETECTION_H
#define PATCH_DETECTION_H

#include <opencv2/opencv.hpp>
#include <list>
#include <iterator>
#include <iostream>
#include <omp.h>

#define N_CORES omp_get_num_threads()

class PatchDetection
{
    public:
        static PatchDetection *instance();
        void process(const cv::Size &_img_size, std::vector<cv::Rect> &_bboxes, const double &_factor);
    protected:
        static PatchDetection *m_instance;
        PatchDetection() {;}
    private:
        static void overlapRoi(const cv::Point &tl1, const cv::Point &tl2,
                               const cv::Size &sz1, const cv::Size &sz2,
                               cv::Rect &_rect);
};

#endif //PATCH_DETECTION_H
