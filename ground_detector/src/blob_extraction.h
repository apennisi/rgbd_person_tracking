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



#ifndef BLOBEXTRACTOR_H
#define BLOBEXTRACTOR_H

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <people_msgs/Clusters.h>
#include <omp.h>

#include "cluster.h"


class BlobExtraction
{
    public:
        BlobExtraction() {;}
        BlobExtraction(const float &_voxel_size, const float &_min_height, const float &_max_height,
                       const float &_theta, const float &_tx, const float &_ty, const bool &_upsidedown,
                       const int &_min_points_cluster, const int &_max_points_cluster,
                       const int &_min_points_subcluster, const int &_max_points_subcluster,
                       const sensor_msgs::CameraInfo &_cameraInfo);
        void process(const std::vector<cv::Point3f> &_cloud, const std::vector<cv::Point3f>& ground3D);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_preprocessing(const std::vector<cv::Point3f> &_cloud);

    public:
        inline std::vector< boost::shared_ptr<Cluster> > clusters() const
        {
            return m_clusters;
        }
        inline people_msgs::Clusters getMessage() const
        {
            return m_msg_clusters;
        }

    private:
        float m_voxel_size;
        float m_double_voxel_size;
        float m_min_height;
        float m_max_height;
        int m_min_points;
        int m_max_points;
        int m_min_points_sub;
        int m_max_points_sub;
        float m_max_distance;
        float m_theta;
        float m_tx;
        float m_ty;
        bool m_upsidedown;
        cv::Mat m_ground;
        people_msgs::Clusters m_msg_clusters;
        pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
        sensor_msgs::CameraInfo m_camera_info;
        std::vector< boost::shared_ptr<Cluster> > m_clusters;
    private:
        void clusterToMsg(const boost::shared_ptr<Cluster> &cl, people_msgs::BasicCluster &_msg);
        void ground(const std::vector<cv::Point3f>& ground3D);
        bool isInside(const cv::Point2f& p);
};

#endif
