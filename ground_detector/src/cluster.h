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



#ifndef CLUSTER_H
#define CLUSTER_H

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>
#include <omp.h>

class Cluster
{
    public:
        Cluster() {;}
        Cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cluster,
                const float &_min_height, const float &_max_height,
                const float &_theta, const float &_tx, const float &_ty,
                const bool &_upsidedown, const int &_min_points_subcluster,
                const int &_max_points_subcluster, const float &_voxel_size,
                const sensor_msgs::CameraInfo &_cameraInfo);
        void process();
        const inline pcl::PointXYZ top() const
        {
            return m_top;
        }
        const inline pcl::PointXYZ getCenter() const
        {
            return m_center;
        }
        const inline pcl::PointXYZ getTopLeft() const
        {
            return m_topLeft;
        }
        const inline pcl::PointXYZ getBottomRight() const
        {
            return m_bottomRight;
        }
        const inline pcl::PointXYZ getFeet() const
        {
            return m_feet;
        }
        const inline cv::Rect getBoundingBox() const
        {
            return m_bounding_box;
        }
        const inline float height() const
        {
            return m_height;
        }
        const inline float width() const
        {
            return m_width;
        }
        const inline float area() const
        {
            return m_area;
        }
        const inline bool morePeople() const
        {
            return m_morepeople;
        }
        const inline pcl::PointCloud<pcl::PointXYZ>::Ptr cluster() const
        {
            return m_cluster;
        }

    private:
        const void computeBBox();
        static void subclustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &_cluster, bool &_morepeople,
                                  const int &_min_points_sub, const int &_max_points_sub, const float &_voxel_size);

    private:
        const inline bool plane() const
        {
            return m_plane;
        }

    private:
        pcl::PointXYZ m_center;
        pcl::PointXYZ m_topLeft;
        pcl::PointXYZ m_bottomRight;
        pcl::PointXYZ m_top;
        pcl::PointXYZ m_bottom;
        pcl::PointXYZ m_right;
        pcl::PointXYZ m_left;
        pcl::PointXYZ m_feet;
        float m_height;
        float m_width;
        float m_min_height;
        float m_max_height;
        float m_theta;
        float m_tx;
        float m_ty;
        float m_area;
        float m_voxel_size;
        int m_upsidedown;
        int m_min_points_sub;
        int m_max_points_sub;
        bool m_plane;
        bool m_morepeople;
        cv::Rect m_bounding_box;
        Eigen::Matrix4f m_rotoTra;
        Eigen::Matrix3f m_K;
        pcl::PointCloud<pcl::PointXYZ>::Ptr m_cluster;
};

#endif
