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


#include "patchmerging.h"

PatchMerging *PatchMerging::m_instance = NULL;

PatchMerging *PatchMerging::instance()
{
    if(!m_instance)
        m_instance = new PatchMerging;
    return m_instance;
}

void PatchMerging::process(const std::vector<cv::Rect> &detections, const std::vector<cv::Rect> &candidates, std::vector<cv::Rect> &people,
                           std::vector<std::pair<int, int> > &indices)
{
    if(detections.size() == 0 || candidates.size() == 0) return;

    int i;
    int mmax;
    int index = -1;
    int index3D = -1;
    int j = 0;

    for(const auto &detection : detections)
    {
        mmax = 0;
        i = 0;
        for(const auto &candidate : candidates)
        {
            cv::Rect rect;
            isOverlapping(candidate.tl(), detection.tl(), candidate.size(), detection.size(), rect);
            if(rect.area() > mmax)
            {
                mmax = rect.area();
                index = j;
                index3D = i;
            }
            ++i;
        }
        if(mmax != 0)
        {
            people.push_back(detection);
            indices.push_back(std::make_pair(index, index3D));
            ++j;
        }
    }
}

void PatchMerging::isOverlapping(const cv::Point &tl1, const cv::Point &tl2, const cv::Size &sz1, const cv::Size &sz2, cv::Rect &_rect)
{
    int x_tl = fmax(tl1.x, tl2.x);
    int y_tl = fmax(tl1.y, tl2.y);
    int x_br = fmin(tl1.x + sz1.width, tl2.x + sz2.width);
    int y_br = fmin(tl1.y + sz1.height, tl2.y + sz2.height);
    if (x_tl < x_br && y_tl < y_br)
    {
        _rect = cv::Rect(cv::Point(x_tl, y_tl), cv::Point(x_br, y_br));
    }
}
