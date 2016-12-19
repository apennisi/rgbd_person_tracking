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


#include "nms.h"

using namespace fpdw;


Nms::Nms(const structs::Nms &_nms)
    : m_type(_nms.type), m_overlap(_nms.overlap)
{
    m_maxn = std::numeric_limits<int>::infinity();
    m_radii = {.15, .15, 1, 1};
    if(_nms.ovrDnm == structs::OvrDnm::FPDW_MIN)
    {
        m_ovrDnm = 0;
    }
    else if(_nms.ovrDnm == structs::OvrDnm::FPDW_UNION)
    {

        m_ovrDnm = 1;
    }
    else
    {
        std::cout << "Error in m_ovrDmn" << std::endl;
        exit(-1);
    }
    m_separate = 0;
    m_thr = -std::numeric_limits<float>::infinity();
}

void Nms::process(std::vector<cv::Rect> &_bboxes, std::vector<float> &_confidences, const cv::Mat &_bbs)
{
    if(_bbs.rows == 0) return;
    cv::Mat column = _bbs.col(4) > m_thr;
    m_bbs = cv::Mat::zeros(cv::Size(5, column.rows), CV_32FC1);

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) shared(column)
    for(uint i = 0; i < column.rows; ++i)
    {
        if(column.at<uchar>(i) == uchar(255))
        {
            m_bbs.at<float>(i, 0) = _bbs.at<float>(i, 0);
            m_bbs.at<float>(i, 1) = _bbs.at<float>(i, 1);
            m_bbs.at<float>(i, 2) = _bbs.at<float>(i, 2);
            m_bbs.at<float>(i, 3) = _bbs.at<float>(i, 3);
            m_bbs.at<float>(i, 4) = _bbs.at<float>(i, 4);
        }
    }

    if(m_bbs.rows == 0)
    {
        return;
    }


    if(m_type == structs::Type::FPDW_MAX)
    {
        std::cout << "1\n";
    }
    else if(m_type == structs::Type::FPDW_MAXG)
    {
        maxg(_bboxes, _confidences);
    }
    else if(m_type == structs::Type::FPDW_MS)
    {
        std::cout << "1\n";
    }
    else if(m_type == structs::Type::FPDW_COVER)
    {

    }
    else
    {
        std::cerr << "Error: Type unknown!" << std::endl;
        exit(-1);
    }


}

void Nms::maxg(std::vector<cv::Rect> &_bboxes, std::vector<float> &_confidences)
{
    cv::Mat indices;
    cv::sortIdx(m_bbs.col(4), indices, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
    int idx;
    cv::Mat bbs(cv::Size(5, m_bbs.rows), CV_32FC1);

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) shared(indices, bbs) private(idx)
    for(uint i = 0; i < indices.rows; ++i)
    {
        idx = indices.at<int>(0, i);
        bbs.at<float>(i, 0) = m_bbs.at<float>(idx, 0);
        bbs.at<float>(i, 1) = m_bbs.at<float>(idx, 1);
        bbs.at<float>(i, 2) = m_bbs.at<float>(idx, 2);
        bbs.at<float>(i, 3) = m_bbs.at<float>(idx, 3);
        bbs.at<float>(i, 4) = m_bbs.at<float>(idx, 4);
    }

    cv::Mat as, xs, xe, ys, ye;

    cv::multiply(bbs.col(2), bbs.col(3), as);
    xs = bbs.col(0);
    xe = bbs.col(0) + bbs.col(2);
    ys = bbs.col(1);
    ye = bbs.col(1) + bbs.col(3);

    std::vector<bool> kp(bbs.rows, true);
    float iw, ih, o, u;

    for(uint i = 0; i < bbs.rows; ++i)
    {
        if(!kp.at(i))
        {
            continue;
        }

        for(uint j = i+1; j < bbs.rows; ++j)
        {
            if(!kp.at(j))
            {
                continue;
            }

            iw = std::min(xe.at<float>(i), xe.at<float>(j)) - std::max(xs.at<float>(i), xs.at<float>(j));

            if(iw <= 0.)
            {
                continue;
            }

            ih = std::min(ye.at<float>(i), ye.at<float>(j)) - std::max(ys.at<float>(i), ys.at<float>(j));

            if(ih <= 0.)
            {
                continue;
            }

            o = iw*ih;
            if(m_ovrDnm)
            {
                u=as.at<float>(i) + as.at<float>(j) - o;
            }
            else
            {
                u=std::min(as.at<float>(i), as.at<float>(j));
            }

            o = o/u;

            if(o > m_overlap)
            {
                kp.at(j) = false;
            }
        }

    }

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
    for(uint i = 0; i < kp.size(); ++i)
    {
        if(kp.at(i))
        {
            #pragma omp critical
            {
                _bboxes.push_back(cv::Rect(bbs.at<float>(i, 0), bbs.at<float>(i, 1),
                                                 bbs.at<float>(i, 2), bbs.at<float>(i, 3)));
                _confidences.push_back(bbs.at<float>(i, 4));
            }
        }
    }

}

