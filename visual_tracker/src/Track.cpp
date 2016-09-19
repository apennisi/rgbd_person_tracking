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


#include <tracker/Track.h>

using namespace tracker;


Track::Track(const cv::Point &_pos, const int &_w, const int &_h, const cv::Scalar &_color)
    : color_(_color)
{
    kalman_ = std::shared_ptr<KalmanFilter>(new KalmanFilter(_pos.x, _pos.y, _w, _h, 0.15));
    lossDetections_ = 0;
    numDetections_ = 1;
    occluded_ = false;
    hide_ = false;
    id_ = -1;
}

const void Track::update(const cv::Point &_pos, const int &_w, const int &_h)
{
    (lossDetections_ > 0) ? --lossDetections_ : lossDetections_ = 0;
    kalman_->correct(_pos.x, _pos.y, _w, _h);
}

const cv::Mat Track::getPosition()
{
    lastKalmanDetection_ = kalman_->predict();
    return lastKalmanDetection_;
}

