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



#include "ground_detection.h"

GroundDetection::GroundDetection(const sensor_msgs::CameraInfo &_camInfo, const float &_theta, const float &_tx, const float &_ty,
                                 const float &_groundThreshold, const float &_max_height, const float &_max_distance,  const bool &_upsidedown)
    : m_theta(_theta), m_tx(_tx), m_ty(_ty), m_groundThreshold(_groundThreshold), m_max_height(_max_height), m_max_distance(_max_distance)
{
    m_theta *= (M_PI / 180);
    m_fx = _camInfo.K.at(0);
    m_fy = _camInfo.K.at(4);
    m_cx = _camInfo.K.at(2);
    m_cy = _camInfo.K.at(5);
    inv_fx = 1. / m_fx;
    inv_fy = 1. / m_fy;

    m_K << _camInfo.K.at(0), _camInfo.K.at(1), _camInfo.K.at(2),
            _camInfo.K.at(3), _camInfo.K.at(4), _camInfo.K.at(5),
            _camInfo.K.at(6), _camInfo.K.at(7), _camInfo.K.at(8);

    m_K_inv = m_K.inverse();

    std::cout << m_K << std::endl;

    std::cout << "rototraslation matrix:" << std::endl;
    m_rotoTra << 1, 0,            0,            m_tx,
               0, cos(m_theta), -sin(m_theta),  m_ty,
               0, sin(m_theta), cos(m_theta),   0, //tz = 0
               0, 0,            0,              1;

    std::cout << m_rotoTra << std::endl;

    m_upsidedown = (_upsidedown) ? -1 : 1;
}

cv::RNG rng(12345);

void GroundDetection::process(const cv::Mat &_depth, const cv::Mat &_nonZeroCoordinates)
{
    m_ground.clear();
    m_candidates.clear();
    m_candidate_points.clear();
    m_ground_points.clear();
    Eigen::Vector4f w;
    cv::Point3f point;
    cv::Point p;
    float d;

#pragma omp parallel for num_threads( omp_get_num_procs() * N_CORES ) private(p, d, point, w)
    for(uint i = 0; i < _nonZeroCoordinates.total(); ++i)
    {
        p = _nonZeroCoordinates.at<cv::Point>(i);
        d = _depth.at<float>(p);
        point.z = d * mm;
        point.x = ((p.x - m_cx) * point.z * inv_fx);
        point.y = m_upsidedown * ((p.y - m_cy) * point.z * inv_fy); //Add "m_upsidedown" because the optical frame has the y-axis upsidedown

        w << point.x, point.y, point.z, 1;
        w = m_rotoTra * w;
        point = cv::Point3f(w(0), w(1), w(2));
        if (point.y <= m_groundThreshold)
        {
            #pragma omp critical
            {
                m_ground.push_back(cv::Point3f(w(0), w(1), w(2)));
                if(m_debug)
                {
                    m_ground_points.push_back(p);
                }
            }
        }
        else if(w(1) <= m_max_height && w(2) <= m_max_distance)
        {
            #pragma omp critical
            {
                m_candidates.push_back(cv::Point3f(w(0), w(1), w(2)));
                if(m_debug)
                {
                    m_candidate_points.push_back(p);
                }
            }
        }
    }

}

