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


#ifndef _PATCHMERGING_H_
#define _PATCHMERGING_H_

#include <opencv2/opencv.hpp>
#include <iostream>

class PatchMerging
{
    public:
        static PatchMerging *instance();
        void process(const std::vector<cv::Rect> &detections, const std::vector<cv::Rect> &candidates,
                     std::vector<cv::Rect> &people, std::vector<std::pair<int, int> >& indices);
    protected:
        PatchMerging() {;}
        static PatchMerging *m_instance;
    private:
        static void isOverlapping(const cv::Point &tl1, const cv::Point &tl2,
                                  const cv::Size &sz1, const cv::Size &sz2,
                                  cv::Rect &_rect);
};

#endif
