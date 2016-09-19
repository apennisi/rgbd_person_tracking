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
#include <iostream>
#include <memory>

namespace tracker
{
    using namespace MyKalmanFilter;
    class Track
    {
        public:
            Track(const cv::Point &_pos, const int &_w, const int &_h, const cv::Scalar &_color);
            const void update(const cv::Point &_pos, const int &_w, const int &_h);
            const cv::Mat getPosition();
            const inline int getId() const
            {
                return id_;
            }
            const inline cv::Scalar getColor() const
            {
                return color_;
            }
            void setWidth(const int& _width)
            {
                width_ = _width;
            }
            void setHeight(const int& _height)
            {
                height_ = _height;
            }

            const inline int width() const
            {
                return width_;
            }

            const inline int height() const
            {
                return height_;
            }

            void addDetection()
            {
                numDetections_++;
            }

            void addLossDetection()
            {
                lossDetections_++;
            }
            void set3DPoint(const cv::Point3d& p)
            {
                Point3D = p;
            }
            const inline cv::Point3d get3DPoint()
            {
                return Point3D;
            }

            const inline int lossDetections()
            {
                kalman_->kf().statePost = lastKalmanDetection_;
                lastDetection_ = cv::Point(lastKalmanDetection_.at<float>(0), lastKalmanDetection_.at<float>(1));
                return lossDetections_;
            }
            const inline int numDetections() const
            {
                return numDetections_;
            }

            const inline void setOccluded(const bool& _occluded)
            {
                occluded_ = _occluded;
            }

            const inline bool occluded() const
            {
                return occluded_;
            }

            void setHide(const bool& _hide)
            {
                hide_ = _hide;
            }
            const inline bool hide() const
            {
                return hide_;
            }

            const inline void reset()
            {
                numDetections_ = 1;
                lossDetections_ = 0;
            }

           const inline void setId(const uint &_id)
           {
               id_ = _id;
           }

           const inline cv::Point lastDetection() const
           {
               return cv::Point(lastDetection_.x + (width_ >> 1), lastDetection_.y + (height_ >> 1));
           }

           const inline void setLastDetection(const cv::Point &p)
           {
               lastDetection_ = p;
           }

           const inline void setHist(const cv::Mat &_hist)
           {
               lastHist_ = _hist.clone();
           }
           const inline cv::Mat hist() const
           {
               return lastHist_;
           }

           const inline void setFeatHist(const cv::Mat &_featHist)
           {
               lastFeatHist_ = _featHist;
           }

           const inline cv::Mat featHist() const
           {
               return lastFeatHist_;
           }

           const inline cv::Mat lastPosition() const
           {
               return lastKalmanDetection_;
           }

           const inline cv::Rect bbox()
           {
               return cv::Rect(lastKalmanDetection_.at<float>(0), lastKalmanDetection_.at<float>(1), lastKalmanDetection_.at<float>(4), lastKalmanDetection_.at<float>(5));
           }

        private:
            std::shared_ptr<KalmanFilter> kalman_;
            int id_;
            int lossDetections_;
            int numDetections_;
            int width_, height_;
            bool occluded_;
            cv::Mat prevHist_;
            cv::Scalar color_;
            bool hide_;
            cv::Point lastDetection_;
            cv::Mat lastKalmanDetection_;
            cv::Mat lastHist_;
            cv::Mat lastFeatHist_;
            cv::Point3d Point3D;
    };
}
