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


#ifndef GROUNDDETECTION_H
#define GROUNDDETECTION_H

#include <opencv2/opencv.hpp>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/CameraInfo.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <numeric>
#include <omp.h>

#define N_CORES omp_get_num_threads()

class GroundDetection
{
    public:
        GroundDetection(const sensor_msgs::CameraInfo &_camInfo, const float &_theta, const float &_tx,
                        const float &_ty, const float &_groundThreshold, const float &_max_height,
                        const float &_max_distance, const bool &_upsidedown);
        void process(const cv::Mat &_depth, const cv::Mat &_nonZeroCoordinates);
        ~GroundDetection() {;}

    public:
        const inline std::vector<cv::Point> image_ground_points() const
        {
            return m_ground_points;
        }
        const inline std::vector<cv::Point3f> ground() const
        {
            return m_ground;
        }
        const inline std::vector<cv::Point3f> no_ground() const
        {
            return m_candidates;
        }
        const inline void setDebug(const bool &_debug)
        {
            m_debug = _debug;
        }
        const inline std::vector<cv::Point> candidate_points() const
        {
            return m_candidate_points;
        }

    private:
        float m_theta;
        float m_tx;
        float m_ty;
        float m_groundThreshold;
        Eigen::Matrix4f m_rotoTra;
        Eigen::Matrix3f m_K;
        Eigen::Matrix3f m_K_inv;
        // Ground plane estimation:
        float m_fx;
        float m_fy;
        float m_cx;
        float m_cy;
        float inv_fx;
        float inv_fy;
        float m_max_height;
        float m_max_distance;
        int m_upsidedown;
        std::vector<cv::Point3f> m_ground;
        std::vector<cv::Point3f> m_candidates;
        std::vector<cv::Point> m_ground_points;
        std::vector<cv::Point> m_candidate_points;
        bool m_debug;

    private:
        const static float mm = 0.001;
};

#endif
