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


#include <tracker/Kalman.h>

using namespace MyKalmanFilter;

//KalmanFilter::KalmanFilter(const int &_x, const int &_y, const float &dt)
KalmanFilter::KalmanFilter(const int &_x, const int &_y, const int &_w, const int &_h, const float &dt)
{
    KF = cv::KalmanFilter(6, 4, 0, CV_32F); //x,y,dx,dy,w,h
    processNoise = cv::Mat(6, 1, CV_32F);

    measurement = cv::Mat_<float>(4, 1, CV_32F);
    measurement.setTo(cv::Scalar(0));


    KF.statePre.at<float>(0, 0) = _x;
    KF.statePre.at<float>(1, 0) = _y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    KF.statePre.at<float>(4) = _w;
    KF.statePre.at<float>(5) = _h;


    KF.statePost.at<float>(0) = _x;
    KF.statePost.at<float>(1) = _y;
    KF.statePost.at<float>(2) = 0;
    KF.statePost.at<float>(3) = 0;
    KF.statePost.at<float>(4) = _w;
    KF.statePost.at<float>(5) = _h;


    KF.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,dt,0,0,0,
                                                     0,1,0,dt,0,0,
                                                     0,0,1,0,0,0,
                                                     0,0,0,1,0,0,
                                                     0,0,0,0,1,0,
                                                     0,0,0,0,0,1);

    KF.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
    KF.measurementMatrix.at<float>(0) = 1.0f;
    KF.measurementMatrix.at<float>(7) = 1.0f;
    KF.measurementMatrix.at<float>(16) = 1.0f;
    KF.measurementMatrix.at<float>(23) = 1.0f;
  

    KF.processNoiseCov=(cv::Mat_<float>(6, 6) <<
                        1e-2, 0, 0, 0, 0, 0,
                        0, 1e-2, 0, 0, 0, 0,
                        0, 0, 2.0f, 0, 0, 0,
                        0, 0, 0, 1.0f, 0, 0,
                        0, 0, 0, 0, 1e-2, 0,
                        0, 0, 0, 0, 0, 1e-2);

    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));

    KF.errorCovPre.at<float>(0) = 1; // px
    KF.errorCovPre.at<float>(7) = 1; // px
    KF.errorCovPre.at<float>(14) = 1;
    KF.errorCovPre.at<float>(21) = 1;
    KF.errorCovPre.at<float>(28) = 1; // px
    KF.errorCovPre.at<float>(35) = 1; // px
}

cv::Mat KalmanFilter::predict()
{
    cv::Mat prediction = KF.predict();

    KF.statePre.copyTo(KF.statePost);
    KF.errorCovPre.copyTo(KF.errorCovPost);

    return prediction;
}

cv::Mat KalmanFilter::correct(const int &_x, const int &_y, const int& _w, const int& _h)
{
    measurement(0) = _x;
    measurement(1) = _y;
    measurement(2) = _w;
    measurement(3) = _h;
    cv::Mat estimated = KF.correct(measurement);
    return estimated;
}
