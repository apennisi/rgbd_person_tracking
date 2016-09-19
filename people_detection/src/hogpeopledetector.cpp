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


#include "hogpeopledetector.h"


HogPeopleDetector::HogPeopleDetector(const double &_hitThreshold, const cv::Size &_winStride,
                               const cv::Size &_padding, const double &_scale,
                               const double &_finalThreshold, const bool &_meanShift)
    : hitThreshold(_hitThreshold), winStride(_winStride), padding(_padding), scale(_scale),
      finalThreshold(_finalThreshold), meanShift(_meanShift)
{
    cv::setNumThreads(cv::getNumberOfCPUs());
    hog = cv::HOGDescriptor(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8,8), cv::Size(8,8), 9, 1, -1,
                            cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
    std::vector<float> detector = cv::HOGDescriptor::getDefaultPeopleDetector();
    hog.setSVMDetector(detector);
    hog.nlevels = 13;
}


void HogPeopleDetector::process(const cv::Mat &img)
{
    found.clear();
    cv::Mat m_img;
    cv::cvtColor(img, m_img, cv::COLOR_BGR2GRAY);
    hog.detectMultiScale(m_img, found, hitThreshold, winStride,
                         padding, scale, finalThreshold, meanShift);
}
