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


#include "patchdetection.h"

PatchDetection* PatchDetection::m_instance = NULL;

PatchDetection *PatchDetection::instance()
{
    if(!m_instance)
    {
        m_instance = new PatchDetection;
    }
    return m_instance;
}

void PatchDetection::process(const cv::Size &_img_size, std::vector<cv::Rect> &_bboxes, const double &_factor)
{
    std::vector<cv::Rect> temp_rects(_bboxes.size());

    int cols = _img_size.width;
    int rows = _img_size.height;


    //creating a bounding
    #pragma omp parallel for num_threads( omp_get_num_procs() * N_CORES ) shared(temp_rects)
    for(int i = 0; i < _bboxes.size(); ++i)
    {
        cv::Rect r = _bboxes[i];
        cv::Point c = cv::Point( (r.tl().x + r.width * .5), (r.tl().y + r.height * .5) );
        cv::Point tl = cv::Point( (c.x - r.width * _factor) < 0 ? 0 : (c.x - r.width * _factor),
                                   (c.y - r.height * _factor) < 0 ? 0 : (c.y - r.height * _factor));


        cv::Point br = cv::Point( (c.x + r.width * _factor) >= cols ? cols - 1 : (c.x + r.width * _factor),
                                  (c.y + r.height * _factor) >= rows ? rows - 1 : (c.y + r.height * _factor));

        temp_rects[i] = cv::Rect(tl, br);
    }

    for(uint i = 0; i < temp_rects.size(); ++i)
    {
        if(temp_rects[i].width < 64 || temp_rects[i].height < 128)
        {
            temp_rects.erase(temp_rects.begin() + i);
        }
    }

    _bboxes.clear();
    _bboxes = temp_rects;
}


void PatchDetection::overlapRoi(const cv::Point &tl1, const cv::Point &tl2,
                                const cv::Size &sz1, const cv::Size &sz2,
                                cv::Rect &_rect)
{
    int x_tl = fmax(tl1.x, tl2.x);
    int y_tl = fmax(tl1.y, tl2.y);
    int x_br = fmin(tl1.x + sz1.width, tl2.x + sz2.width);
    int y_br = fmin(tl1.y + sz1.height, tl2.y + sz2.height);
    if (x_tl < x_br && y_tl < y_br)
    {
        x_tl = fmin(tl1.x, tl2.x);
        y_tl = fmin(tl1.y, tl2.y);
        x_br = fmax(tl1.x + sz1.width, tl2.x + sz2.width);
        y_br = fmax(tl1.y + sz1.height, tl2.y + sz2.height);

        _rect = cv::Rect(cv::Point(x_tl, y_tl), cv::Point(x_br, y_br));
    }
}


