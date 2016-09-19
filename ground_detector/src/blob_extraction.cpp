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


#include "blob_extraction.h"


BlobExtraction::BlobExtraction(const float &_voxel_size, const float &_min_height, const float &_max_height,
                               const float &_theta, const float &_tx, const float &_ty, const bool &_upsidedown,
                               const int &_min_points_cluster, const int &_max_points_cluster,
                               const int &_min_points_subcluster, const int &_max_points_subcluster,
                               const sensor_msgs::CameraInfo &_cameraInfo)
    : m_voxel_size(_voxel_size), m_min_height(_min_height), m_max_height(_max_height),
      m_theta(_theta), m_tx(_tx), m_ty(_ty), m_upsidedown(_upsidedown), m_min_points(_min_points_cluster),
      m_max_points(_max_points_cluster), m_min_points_sub(_min_points_subcluster), m_max_points_sub(_max_points_subcluster),
      m_camera_info(_cameraInfo)
{
    m_double_voxel_size = 2 * m_voxel_size;
}



pcl::PointCloud<pcl::PointXYZ>::Ptr BlobExtraction::cloud_preprocessing(const std::vector<cv::Point3f> &_cloud)
{
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
   cv::Point3f p;
   cloud_downsampled->points.resize(_cloud.size());

#pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) private(p) shared(cloud_downsampled)
   for(uint i = 0; i < _cloud.size(); ++i)
   {
       p = _cloud.at(i);
       cloud_downsampled->points.at(i) = pcl::PointXYZ(p.x, p.y, p.z);
   }

   // Voxel grid filtering:
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter_object;
   voxel_grid_filter_object.setInputCloud(cloud_downsampled);

   voxel_grid_filter_object.setLeafSize (m_voxel_size, m_voxel_size, m_voxel_size);
   voxel_grid_filter_object.setFilterFieldName("z");
   voxel_grid_filter_object.filter (*cloud_filtered);

   return cloud_filtered;
}


void BlobExtraction::process(const std::vector<cv::Point3f> &_cloud, const std::vector<cv::Point3f>& ground3D)
{
    if(_cloud.size() > 0)
    {
        m_clusters.clear();
        m_msg_clusters.clusters.clear();
        m_cloud = cloud_preprocessing(_cloud);
        ground(ground3D);

        // Euclidean Clustering:
        std::vector<pcl::PointIndices> cluster_indices;
        typename pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(m_cloud);
        pcl::EuclideanClusterExtraction<pcl::PointXYZ > ec;
        ec.setClusterTolerance(m_double_voxel_size);
        ec.setMinClusterSize(m_min_points);
        ec.setMaxClusterSize(m_max_points);
        ec.setSearchMethod(tree);
        ec.setInputCloud(m_cloud);
        ec.extract(cluster_indices);

        people_msgs::BasicCluster basicCluster;

#pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
        for(uint i = 0; i < cluster_indices.size(); ++i)
        //for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator pit = cluster_indices.at(i).indices.begin (); pit != cluster_indices.at(i).indices.end (); ++pit)
            {
                cloud_cluster->points.push_back (m_cloud->points[*pit]);
            }
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            boost::shared_ptr<Cluster> cl(new Cluster(cloud_cluster, m_min_height, m_max_height, m_theta,
                                                      m_tx, m_ty, m_upsidedown, m_min_points_sub, m_max_points_sub,
                                                      m_voxel_size, m_camera_info));
            cl->process();
            if(cl->height() < m_min_height)
            {
                cl.reset();
            }
            else
            {
                #pragma omp critical
                {
                    /*if(isInside(cv::Point2f(cl->getCenter().x, cl->getCenter().z)))
                    {*/
                        clusterToMsg(cl, basicCluster);
                        m_msg_clusters.clusters.push_back(basicCluster);
                        m_clusters.push_back(cl);
                    //}
                }
            }

        }

    }


}

void BlobExtraction::clusterToMsg(const boost::shared_ptr<Cluster> &cl, people_msgs::BasicCluster &_msg)
{
    _msg.area = cl->area();
    _msg.bottomRight.x = cl->getBottomRight().x;
    _msg.bottomRight.y = cl->getBottomRight().y;
    _msg.bottomRight.z = cl->getBottomRight().z;
    _msg.boundingBox.topLeft.x = cl->getBoundingBox().tl().x;
    _msg.boundingBox.topLeft.y = cl->getBoundingBox().tl().y;
    _msg.boundingBox.bottomRight.x = cl->getBoundingBox().br().x;
    _msg.boundingBox.bottomRight.y = cl->getBoundingBox().br().y;
    _msg.boundingBox.width = cl->getBoundingBox().width;
    _msg.boundingBox.height = cl->getBoundingBox().height;
    _msg.center.x = cl->getCenter().x;
    _msg.center.y = cl->getCenter().y;
    _msg.center.z = cl->getCenter().z;
    _msg.distance = _msg.center.z;
    _msg.feet.x = cl->getFeet().x;
    _msg.feet.y = cl->getFeet().y;
    _msg.feet.z = cl->getFeet().z;
    _msg.height = cl->height();
    _msg.topLeft.x = cl->getTopLeft().x;
    _msg.topLeft.y = cl->getTopLeft().y;
    _msg.topLeft.z = cl->getTopLeft().z;
    _msg.width = cl->width();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::toPCLPointCloud2(*(cl->cluster()), pcl_pc2);
    pcl_conversions::fromPCL(pcl_pc2, _msg.cluster);
}

void BlobExtraction::ground(const std::vector<cv::Point3f> &ground3D)
{
    m_ground = cv::Mat(cv::Size(601, 701), CV_8UC1, cv::Scalar(0));

    for(uint i = 0; i <ground3D.size(); ++i)
    {
        cv::Point p((ground3D.at(i).x + 3)*100, ground3D.at(i).z*100);
        if(p.x >= 0 && p.x <= 600 && p.y >= 0 && p.y <= 700 )
        {
            cv::circle(m_ground, p, 1, cv::Scalar(100), -1);
        }
    }

    cv::dilate(m_ground, m_ground, cv::Mat());
    cv::rectangle(m_ground, cv::Rect(0, 0, m_ground.cols, 210), cv::Scalar(100), -1);
}

bool BlobExtraction::isInside(const cv::Point2f& _p)
{
    cv::Point p((_p.x+3)*100, _p.y*100);
    const int range = 40;

    cv::Mat temp = m_ground(cv::Rect(p.x - (range >> 1), p.y - (range >> 1), range, range));


    int nonZero = cv::countNonZero(temp);
    cv::circle(m_ground, p, 4, cv::Scalar(255), -1);
    float perc = nonZero / float(temp.total());
    cv::rectangle(m_ground, cv::Rect(p.x - (range >> 1), p.y - (range >> 1), range, range), cv::Scalar(255), 1);
    cv::imshow("m_ground", m_ground);

    cv::waitKey(1);
    return perc > 0.4;
}
