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


//ROS
#include <ros/ros.h>
#include <people_msgs/SegmentedImage.h>
#include <cv_bridge/cv_bridge.h>
//C++
#include <iostream>
//OPENCV
#include <opencv2/opencv.hpp>

//PEOPLE DETECTOR
#include "fpdw_detector.h"
#ifdef HOG
#include "hogpeopledetector.h"
#endif

//PATCH
#ifdef HOG
#include "patchdetection.h"
#endif
#include "patchmerging.h"

//PARAMETERS
std::string _dataset_file;
double _confidence;
double _imageScalingFactor;

//VARIABLES
ros::Subscriber people_sub;
ros::Publisher people_detected;
std::shared_ptr<fpdw::detector::FPDWDetector> detector;
#ifdef HOG
std::shared_ptr<HogPeopleDetector> hog_detector;
#endif
cv::Mat image, resized;
cv_bridge::CvImageConstPtr rgbImage;

int counter;
std::stringstream ss;

///TODO PATCH MERGING ACCORDING TO THRESHOLD

void people_detection_callback(const people_msgs::SegmentedImageConstPtr &_candidates)
{
    const people_msgs::SegmentedImage &candidates = *_candidates;

    try
    {
        rgbImage = cv_bridge::toCvCopy(candidates.image, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception &ex)
    {
        ROS_ERROR("cv_bridge RGB exception: %s", ex.what());
        return;
    }

    image = rgbImage->image.clone();
    std::vector<cv::Rect> bboxes, cts, scaledBBoxes;
    cv::Rect r;
#ifndef HOG
    try
    {
        cv::resize(image, resized, cv::Size(image.cols/_imageScalingFactor, image.rows/_imageScalingFactor));
        detector->process(resized);
        bboxes = detector->getBBoxes();
    }
    catch(...)
    {
        ROS_ERROR("PROBLEM WITH THE DETECTOR!");
        exit(-1);
    }

    for(const auto &j : bboxes)
    {
        r = cv::Rect(j.tl().x * _imageScalingFactor, j.tl().y * _imageScalingFactor, j.width * _imageScalingFactor, j.height * _imageScalingFactor);
        scaledBBoxes.push_back(r);
        cv::rectangle(image, r, cv::Scalar(255, 0, 0), 2);
    }

    for(const auto &i : candidates.clusters.clusters)
    {
        r = cv::Rect(i.boundingBox.topLeft.x, i.boundingBox.topLeft.y, i.boundingBox.width, i.boundingBox.height);
        cv::rectangle(image, r, cv::Scalar(0, 255, 0), 2);
        cts.push_back(r);
    }
    std::vector<cv::Rect> people;
    std::vector<std::pair<int, int> > indices;
    PatchMerging::instance()->process(scaledBBoxes, cts, people, indices);
    people_msgs::SegmentedImage selected_people;
    selected_people.image = candidates.image;

    people_msgs::BasicCluster _msg;
    for(const auto &idx : indices)
    {
        _msg = candidates.clusters.clusters.at(idx.second);
        r = people.at(idx.first);
        _msg.boundingBox.topLeft.x = r.x;
        _msg.boundingBox.topLeft.y = r.y;
        _msg.boundingBox.width = r.width;
        _msg.boundingBox.height = r.height;
        selected_people.clusters.clusters.push_back(_msg);
        cv::rectangle(image, r, cv::Scalar(0, 0, 255), 2);
    }

    people_detected.publish(selected_people);

#else
    PatchDetection::instance()->process(image.size(), cts, 1.25);

    cv::Mat patch;
    int x, y, w, h;
    for(const auto &bbox : cts)
    {

        patch = image(bbox);
        try
        {
            hog_detector->process(patch);
            bboxes = hog_detector->getRectangle();
        }
        catch(...)
        {
            ROS_ERROR("PROBLEM WITH THE DETECTOR!");
        }

        for(const auto &j : bboxes)
        {
            x = bbox.tl().x + j.tl().x;
            x = x > 0 ? x : 0;
            y = bbox.tl().y + j.tl().y;
            y = y > 0 ? y : 0;
            w = x + j.width;
            w = w > image.cols ? j.width - (w - image.cols) : j.width;
            h = y + j.height;
            h = h > image.rows ? j.height - (h - image.rows) : j.height;
            cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
        }
    }
#endif
    cv::imshow("image", image);
    cv::waitKey(1);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "people_detector_node", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");
    
    nh.param("dataset", _dataset_file, std::string("/home/morpheus/workspace/ros/catkin/src/people_tracker/people_detection/config/inria_detector.xml"));
    nh.param("confidence", _confidence, double(60.));
    nh.param("image_scaling_factor", _imageScalingFactor, double(1.5));

    //set the detectorJune 12
    detector = std::shared_ptr<fpdw::detector::FPDWDetector>(new fpdw::detector::FPDWDetector(_dataset_file, _confidence));
#ifdef HOG
    hog_detector = std::shared_ptr<HogPeopleDetector>(new HogPeopleDetector(0.5, cv::Size(4, 4), cv::Size(32, 32), 1.2, 2, false));
#endif

    people_sub = nh.subscribe("/bodyclusters", 1, people_detection_callback);
    people_detected = nh.advertise<people_msgs::SegmentedImage>("/people_detected", 1);

	ros::spin();
}
