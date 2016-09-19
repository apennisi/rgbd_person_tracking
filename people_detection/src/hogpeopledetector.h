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


#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

class HogPeopleDetector
{
    public:
        HogPeopleDetector(const double &_hitThreshold, const cv::Size &_winStride,
                       const cv::Size &_padding, const double &_scale,
                       const double &_finalThreshold, const bool &_meanShift);
        HogPeopleDetector() {;}
        void process(const cv::Mat &img);
        const inline std::vector<cv::Rect> getRectangle() const&
        {
            return found;
        }
    private:
        cv::HOGDescriptor hog;
        double hitThreshold;
        cv::Size winStride;
        cv::Size padding;
        double scale;
        double finalThreshold;
        bool meanShift;
        std::vector<cv::Rect> found, found_filtered;
};
