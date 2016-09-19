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



#include <opencv2/opencv.hpp>

#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <ros/node_handle.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <people_msgs/SegmentedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "ground_detection.h"
#include "blob_extraction.h"
#include "cluster.h"

#include <sensor_msgs/LaserScan.h>

//PARAMETERS
sensor_msgs::CameraInfo cameraInfo;
double _theta;
double _tx, _ty, _tz;
double _groundThreshold;
double _voxel_size;
double _min_height;
double _max_height;
double _max_distance;
bool _debug;
bool _upSideDownYAxis;
bool _apply_denoising;
int _min_points_cluster;
int _max_points_cluster;
int _min_points_subcluster;
int _max_points_subcluster;
std::string _depth_topic;
std::string _rgb_topic;
std::string _camera_info_topic;

//VARIABLES
boost::shared_ptr<GroundDetection> ground_detector;
boost::shared_ptr<BlobExtraction> blob_extractor;
cv_bridge::CvImage rgb;
people_msgs::SegmentedImage segImg;

//PUBLISHER
ros::Publisher clusters_sub;

//DEBUGGING VARIABLES
std::stringstream text;
int fontFace = cv::FONT_HERSHEY_SIMPLEX;
double fontScale = 0.5;
int thickness = 2;
ros::Publisher rgbdepth_pub, boundingbox_pub;

void groundfloor_callback(const sensor_msgs::ImageConstPtr& _depthImage, const sensor_msgs::ImageConstPtr& _rgbImage)
{
    cv_bridge::CvImageConstPtr rgbImage, depthRaw, depthImage;

    try
    {
        rgbImage = cv_bridge::toCvCopy(_rgbImage, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception &ex)
    {
        ROS_ERROR("cv_bridge RGB exception: %s", ex.what());
        return;
    }

    try
    {
        depthImage = cv_bridge::toCvShare(_depthImage, sensor_msgs::image_encodings::TYPE_32FC1);
        depthRaw = cv_bridge::toCvShare(_depthImage, sensor_msgs::image_encodings::TYPE_8UC1);
    }
    catch(cv_bridge::Exception &ex)
    {
        ROS_ERROR("cv_bridge DEPTH exception: %s", ex.what());
        return;
    }

    cv::Mat nonZeroCoordinates;
    cv::findNonZero(depthRaw->image, nonZeroCoordinates);

    ground_detector->process(depthImage->image, nonZeroCoordinates);
    const std::vector<cv::Point3f> &no_ground = ground_detector->no_ground();
    const std::vector<cv::Point3f> &ground3D = ground_detector->ground();
    blob_extractor->process(no_ground, ground3D);

    //PUBLISH THE SEGMENTED IMAGE
    segImg.clusters = blob_extractor->getMessage();
    segImg.image = *_rgbImage;
    clusters_sub.publish(segImg);

    if(_debug)
    {
        const std::vector< boost::shared_ptr<Cluster> > &clusters = blob_extractor->clusters();
        const std::vector<cv::Point> &ground = ground_detector->image_ground_points();
        cv::Mat rgb = rgbImage->image.clone();

        const std::vector<cv::Point> &candidates = ground_detector->candidate_points();
        cv::Point p;
#pragma omp parallel for num_threads( omp_get_num_procs() * N_CORES ) shared(rgb)
        for(uint i = 0; i < candidates.size(); ++i)
        {
            p = candidates.at(i);
            rgb.at<cv::Vec3b>(p)  = cv::Vec3b(0, 0, 255);
        }
#pragma omp parallel for num_threads( omp_get_num_procs() * N_CORES ) shared(rgb)
        for(uint i = 0; i < ground.size(); ++i)
        {
            p = ground.at(i);
            rgb.at<cv::Vec3b>(p)  = cv::Vec3b(255, 0, 0);
        }

        cv::Mat bbox = rgbImage->image.clone();
        boost::shared_ptr<Cluster> cl;
        cv::Point point;
#pragma omp parallel for num_threads( omp_get_num_procs() * N_CORES ) shared(bbox) private(text, cl, point)
        for(uint i = 0; i < clusters.size(); ++i)
        {
            cl = clusters.at(i);
            text.str("");
            text << "Distance: " << cl->getCenter().z;
            point = cv::Point((cl->getBoundingBox().br().x + cl->getBoundingBox().x ) * 0.5 - 10, (cl->getBoundingBox().br().y + cl->getBoundingBox().y ) *0.5);
            #pragma omp critical
            {
                cv::rectangle(bbox, cl->getBoundingBox(), cv::Scalar(0, 255, 0), 2);
                cv::putText(bbox, text.str(), point, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness, 8);
            }
        }

        cv_bridge::CvImage rgbdepth, boundingbox;
        rgbdepth.header   = rgbImage->header;
        rgbdepth.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        rgbdepth.image    = rgb;

        boundingbox.header   = rgbImage->header;
        boundingbox.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        boundingbox.image    = bbox;

        rgbdepth_pub.publish(rgbdepth.toImageMsg());
        boundingbox_pub.publish(boundingbox.toImageMsg());

    }

}

void print_paramenters()
{
    std::cout << "PARAMETERS" << std::endl;
    std::cout << std::endl;
    std::cout << "theta:\t" << _theta << std::endl;
    std::cout << "tx:\t" << _tx << std::endl;
    std::cout << "ty:\t" << _ty << std::endl;
    std::cout << "groundThreshold:\t" << _groundThreshold << std::endl;
    std::cout << "voxel_size:\t" << _voxel_size << std::endl;
    std::cout << "min_height:\t" << _min_height << std::endl;
    std::cout << "max_height:\t" << _max_height << std::endl;
    std::cout << "max_distance:\t" << _max_distance << std::endl;
    std::cout << "debug:\t" << _debug << std::endl;
    std::cout << "depth_topic:\t" << _depth_topic << std::endl;
    std::cout << "rgb_topic:\t" << _rgb_topic << std::endl;
    std::cout << "camera_info_topic:\t" << _camera_info_topic << std::endl;

    if(_debug)
        std::cout << "***DEBUGGING MODE***" << std::endl;

}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "ground_detector_node", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    nh.param("theta", _theta, double(12));
    nh.param("tx", _tx, double(0));
    nh.param("ty", _ty, double(1.5));
    nh.param("tz", _tz, double(0));
    nh.param("groundThreshold", _groundThreshold, double(0.05)); //default 5cm
    nh.param("voxel_size", _voxel_size, double(0.06)); //default 6 cm
    nh.param("min_height", _min_height, double(1.0)); //default 1.3m
    nh.param("max_height", _max_height, double(2.0)); //default 2m
    nh.param("max_distance", _max_distance, double(5.)); //detection rate in meters
    nh.param("up_side_down", _upSideDownYAxis, true); //if the Y-axis is upside down
    nh.param("min_point_cluster", _min_points_cluster, int(100)); //min number of points to made a cluster
    nh.param("max_point_cluster", _max_points_cluster, int(10000)); //min number of points to made a cluster
    nh.param("min_point_subcluster", _min_points_subcluster, int(200)); //min number of points to made a cluster
    nh.param("max_point_subcluster", _max_points_subcluster, int(10000)); //min number of points to made a cluster
    nh.param("debug", _debug, false);
    nh.param("depth_topic", _depth_topic, std::string("/top_camera/depth/image_raw"));
    nh.param("rgb_topic", _rgb_topic, std::string("/top_camera/rgb/image_raw"));
    nh.param("camera_info_topic", _camera_info_topic, std::string("/top_camera/depth/camera_info"));

    print_paramenters();

    //CAMERA INFO
    try
    {
        ROS_INFO("WAITING FOR CAMERA INFO");
        cameraInfo = *(ros::topic::waitForMessage<sensor_msgs::CameraInfo>(_camera_info_topic, nh));
        ground_detector = boost::shared_ptr<GroundDetection>(new GroundDetection(cameraInfo, _theta, _tx, _ty, _groundThreshold, _max_height, _max_distance, _upSideDownYAxis));
        ground_detector->setDebug(_debug);
        ROS_INFO("GOT CAMERA INFO");
    }
    catch(std::exception ex)
    {
        ROS_WARN("NO CAMERA INFO");
    }

    //BLOB EXTRACTOR
    blob_extractor = boost::shared_ptr<BlobExtraction>(new BlobExtraction(_voxel_size, _min_height, _max_height, _theta, _tx, _ty, _upSideDownYAxis,
                                                                          _min_points_cluster, _max_points_cluster, _min_points_subcluster,
                                                                          _max_points_subcluster, cameraInfo));

    //MESAGGE SYNCHRONIZATION
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, _depth_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, _rgb_topic, 1);


    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), depth_sub, rgb_sub);

    sync.registerCallback(boost::bind(&groundfloor_callback, _1, _2));

    clusters_sub = nh.advertise<people_msgs::SegmentedImage>("/rgbdsegmented", 1);

    if(_debug)
    {
        rgbdepth_pub = nh.advertise<sensor_msgs::Image>("/rgbdepth", 1);
        boundingbox_pub = nh.advertise<sensor_msgs::Image>("/boundingbox", 1);
    }

    ros::spin();

}
