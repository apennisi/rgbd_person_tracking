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



#include <ros/ros.h>
#include <people_msgs/SegmentedImage.h>
#include <iostream>

//PARAMETERS
double _min;

//VARIABLES
ros::Subscriber dispatcher_sub;
ros::Publisher dispatcher_face_pub;
ros::Publisher dispatcher_body_pub;

void dispatcher_callback(const people_msgs::SegmentedImageConstPtr &_segmentedCluster)
{
    const people_msgs::SegmentedImage &clusters = *_segmentedCluster;

    people_msgs::SegmentedImage faces, bodies;
    faces.image = clusters.image;
    bodies.image = clusters.image;

    for(const auto &i : clusters.clusters.clusters)
    {
        if(i.distance <= _min)
        {
            faces.clusters.clusters.push_back(i);
        }
        else
        {
            bodies.clusters.clusters.push_back(i);
        }
    }

    dispatcher_body_pub.publish(bodies);
    dispatcher_face_pub.publish(faces);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "dispatcher_node", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    nh.param("min", _min, double(1.2));

    dispatcher_sub = nh.subscribe("/rgbdsegmented", 1, dispatcher_callback);
    dispatcher_face_pub = nh.advertise<people_msgs::SegmentedImage>("/faceclusters", 1);
    dispatcher_body_pub = nh.advertise<people_msgs::SegmentedImage>("/bodyclusters", 1);
    ros::spin();
}
