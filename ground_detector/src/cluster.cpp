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



#include "cluster.h"


Cluster::Cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cluster,
                 const float &_min_height, const float &_max_height,
                 const float &_theta, const float &_tx, const float &_ty,
                 const bool &_upsidedown, const int &_min_points_subcluster,
                 const int &_max_points_subcluster, const float &_voxel_size,
                 const sensor_msgs::CameraInfo &_cameraInfo)
    : m_cluster(_cluster), m_min_height(_min_height), m_max_height(_max_height),
      m_theta(_theta), m_tx(_tx),
      m_ty(_ty), m_voxel_size(_voxel_size),
      m_min_points_sub(_min_points_subcluster), m_max_points_sub(_max_points_subcluster)
{
    m_theta *= (M_PI / 180);

    m_K << _cameraInfo.K.at(0), _cameraInfo.K.at(1), _cameraInfo.K.at(2),
            _cameraInfo.K.at(3), _cameraInfo.K.at(4), _cameraInfo.K.at(5),
            _cameraInfo.K.at(6), _cameraInfo.K.at(7), _cameraInfo.K.at(8);

    Eigen::Matrix4f temp_rotoTra;
    temp_rotoTra << 1, 0,            0,            m_tx,
               0, cos(m_theta), -sin(m_theta),  m_ty,
               0, sin(m_theta), cos(m_theta),   0, //tz = 0
               0, 0,            0,              1;

    m_rotoTra = temp_rotoTra.inverse();

    m_upsidedown = (_upsidedown) ? -1 : 1;
}

void Cluster::process()
{
    m_top =
        m_bottom =
        m_left =
        m_right = m_cluster->points[0];
    m_center = m_cluster->points[0];

    pcl::PointXYZ p;
#pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) private(p)
    for(uint i = 1; i < m_cluster->points.size(); ++i)
    {
        p = m_cluster->points.at(i);

        if(p.x < m_left.x)
        #pragma omp critical
        {
            m_left = p;
        }
        else if(p.x > m_right.x)
        #pragma omp critical
        {
            m_right = p;
        }

        if(p.y > m_top.y)
        #pragma omp critical
        {
            m_top = p;
        }
        else if(p.y < m_bottom.y)
        #pragma omp critical
        {
            m_bottom = p;
        }

        m_center.x += p.x;
        m_center.y += p.y;
        m_center.z += p.z;
    }

    m_center.x /= m_cluster->points.size();
    m_center.y /= m_cluster->points.size();
    m_center.z /= m_cluster->points.size();

    m_height = m_top.y - m_bottom.y;
    m_width = std::max(m_right.x, m_left.x) - std::min(m_right.x, m_left.x);
    m_area = m_height * m_width;

    m_bottomRight.x = m_right.x;
    m_bottomRight.y = m_bottom.y;

    m_topLeft.x = m_left.x;
    m_topLeft.y = m_top.y;

    m_feet = m_bottom;

    computeBBox();

    //subclusterthread.join();
}

void Cluster::subclustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cluster, bool &_morepeople,
                            const int &_min_points_sub, const int &_max_points_sub, const float &_voxel_size)
{
    std::vector<pcl::PointIndices> cluster_indices;
    typename pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(_cluster);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ > ec;
    ec.setClusterTolerance(_voxel_size);
    ec.setMinClusterSize(_min_points_sub);
    ec.setMaxClusterSize(_max_points_sub);
    ec.setSearchMethod(tree);
    ec.setInputCloud(_cluster);
    ec.extract(cluster_indices);

    _morepeople = cluster_indices.size() > 1;
}

const void Cluster::computeBBox()
{
    Eigen::Vector4f w_top, w_bottom, w_left, w_right;

    w_top << m_top.x, m_top.y, m_top.z, 1;
    w_bottom << m_bottom.x, m_bottom.y, m_bottom.z, 1;
    w_left << m_left.x, m_left.y, m_left.z, 1;
    w_right << m_right.x, m_right.y, m_right.z, 1;

    w_top = m_rotoTra * w_top;
    w_bottom = m_rotoTra * w_bottom;
    w_left = m_rotoTra * w_left;
    w_right = m_rotoTra * w_right;

    Eigen::Vector3f w_top_, w_bottom_, w_left_, w_right_;

    w_top_ << w_top(0), w_top(1), w_top(2);
    w_bottom_ << w_bottom(0), w_bottom(1), w_bottom(2);
    w_left_ << w_left(0), w_left(1), w_left(2);
    w_right_ << w_right(0), w_right(1), w_right(2);

    w_top_(1) *= m_upsidedown; //Add "m_upsidedown" because the optical frame has the y-axis upsidedown
    w_bottom_(1) *= m_upsidedown; //Add "m_upsidedown" because the optical frame has the y-axis upsidedown
    w_left_(1) *= m_upsidedown; //Add "m_upsidedown" because the optical frame has the y-axis upsidedown
    w_right_(1) *= m_upsidedown; //Add "m_upsidedown" because the optical frame has the y-axis upsidedown

    w_top_ = m_K * w_top_;
    w_bottom_ = m_K * w_bottom_;
    w_left_ = m_K * w_left_;
    w_right_ = m_K * w_right_;

    w_top_ /= w_top_(2);
    w_bottom_ /= w_bottom_(2);
    w_left_ /= w_left_(2);
    w_right_ /= w_right_(2);

    m_bounding_box = cv::Rect(cv::Point2i(w_left_(0), w_top_(1)),
                              cv::Point2i(w_right_(0),  w_bottom_(1)));
}
