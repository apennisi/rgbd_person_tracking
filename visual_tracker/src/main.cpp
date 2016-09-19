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

//VISUAL TRACKER
#include <tracker/Tracker.h>

//TRACKER
tracker::Tracker tr;
//SUBSCRIBER
ros::Subscriber detections;
//PUBLISHER
ros::Publisher tracks;
//VARIABLES
std::vector<tracker::detection_struct> rects;
cv::Mat frame, resize;
cv_bridge::CvImageConstPtr rgbImage;
cv::Rect r;
double _imageScalingFactor;

void people_tracking(const people_msgs::SegmentedImageConstPtr &_detections)
{
    const people_msgs::SegmentedImage &detections = *_detections;

    try
    {
        rgbImage = cv_bridge::toCvCopy(detections.image, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception &ex)
    {
        ROS_ERROR("cv_bridge RGB exception: %s", ex.what());
        return;
    }

    frame = rgbImage->image.clone();
    cv::resize(frame, resize, cv::Size(frame.cols/_imageScalingFactor, frame.rows/_imageScalingFactor));

    rects.clear();
    for(const auto &det : detections.clusters.clusters)
    {
        tracker::detection_struct d;
        d.bbox = cv::Rect(det.boundingBox.topLeft.x/_imageScalingFactor, det.boundingBox.topLeft.y/_imageScalingFactor,
                          det.boundingBox.width/_imageScalingFactor, det.boundingBox.height/_imageScalingFactor);
        d.point3D = cv::Point3d(det.center.x, det.center.y, det.center.z);
        rects.push_back(d);
    }

    tr.track(rects, resize);
    tr.visualize(frame, _imageScalingFactor);
    people_msgs::Tracks t;
    tr.generateMessage(t, _imageScalingFactor);
    tracks.publish(t);

    cv::imshow("Tracker", frame);
    cv::waitKey(1);

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "visual_tracker", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");
    nh.param("image_scaling_factor", _imageScalingFactor, double(1.5));
    detections = nh.subscribe("/people_detected", 1, people_tracking);
    tracks = nh.advertise<people_msgs::Tracks>("/tracks", 1);

    ros::spin();

    return 0;
}
