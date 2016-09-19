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


#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <adaboost/StrongClassifier.hpp>
#include <adaboost/KalmanWeakClassifier.hpp>
#include <adaboost/ColorFeature.hpp>
#include <adaboost/ColorOrbFeature.hpp>
#include <tracker/Track.h>
#include <omp.h>
#include <thread>
#include <functional>
#include <mutex>
#include <nonfree/features2d.hpp>
#include <people_msgs/Tracks.h>

namespace tracker
{
    #define BINS 16		  	// number of histogram bins for every dimension
    #define CHANNELS 3		// number of histogram dimensions (equal to the number of image channels)

    struct detection_struct
    {
        cv::Rect bbox;
        cv::Point3d point3D;
    };

    //typedef adaboost::ColorFeature_<BINS, CHANNELS> _ColorFeature;
    typedef adaboost::ColorOrbFeature_<BINS, CHANNELS> _ColorOrbFeature;
    //typedef adaboost::KalmanWeakClassifier<_ColorFeature> _ColorWeakClassifier;
    typedef adaboost::KalmanWeakClassifier<_ColorOrbFeature> _ColorWeakClassifier;
    enum colorspace{RGB, HSV, LUV, LAB};

    class Tracker
    {
        public:
            /**
             * @brief Classifier
             * @param _classifiers number of weak classifiers considered at each iteration
             * @param _selectors number of weak classifier that compose the strong classifier
             * @param _numFeatureToDraw number of chosen features whose mean value is drawn for visualization
             */
            Tracker(const int &_classifiers = 250, const int &_selectors = 100, const int &_numFeatureToDraw = 200,
                    const float &_assocThres = 0.5, const colorspace &_color = colorspace::LAB, const int& _accDetection = 5,
                    const int& _lossDetection = 5);
            void track(const std::vector<detection_struct> & _detections,
                       const cv::Mat &_image);
            void visualize(cv::Mat &_image, const double& scaling_factor);
            void generateMessage(people_msgs::Tracks& _msg, const double& scaling_factor);
            virtual ~Tracker();

        private:
            int classifiers_;
            int selectors_;
            int numFeaturesToDraw_;
            int idCounter_;
            int accDetection_;
            int lossDetection_;
            bool initialize_;
            int cZeroDet_;
            float assocThres_;
            colorspace color_;
            cv::Mat image_;
            cv::Mat keyPointImage_;
            cv::Mat keyPointMask_;
            cv::Mat descriptors_;//in caso cancellare
            std::vector<cv::KeyPoint> keypoints_;//in caso cancellare
            cv::RNG rng;
            std::vector<detection_struct> detections_;
            std::vector<std::shared_ptr<adaboost::StrongClassifier> > strongClassifiers_;
            std::vector<std::shared_ptr<Track> > tracks_;
            std::vector<cv::Mat> negMasks_;
            std::vector<cv::Mat> histograms_;
            std::vector<cv::Mat> featureHistograms_;
            std::map<int, std::pair<int, float> > oldTracks; //ID_CLASSIFIER, PAIR<ID_DETECTION, CONFIDENCE>
            std::map<int, std::set<int> > occlusionMap; //ID_TRACK, VECTOR OF OCCLUSIONS
            std::vector<int> newTracks;
            std::vector<int> classifiersToDelete;
        private:
            /**
             * @brief calcColorHistogram
             * @param image
             * @param blobMask
             * @param bins
             * @param channels
             * @param colorspace
             * @param histogram
             */
            static void calcColorHistogram(const cv::Mat& _image, const cv::Mat& _blobMask, const int& _bins, const int& _channels,
                                    cv::Mat& _histogram);
            /**
             * @brief createNegativeHistograms
             * @param _image
             * @param _mask
             * @param _n
             * @param _bins
             * @param _negative_histograms
             * @param _detections
             * @param det
             */
            void createNegativeHistograms(const cv::Mat& _image, const cv::Mat& _mask, const int& _n, const int& _bins, std::vector<cv::Mat>& _negative_histograms,
                                          std::vector<cv::Mat>& _negative_feature_histograms, const std::vector<detection_struct>& _detections, const int &det);

            static void calcFeatures(const cv::Rect& _detection, cv::Mat& _histrogram, const cv::Mat& _descriptors,
                                     const cv::Mat& _keyPointImage, const cv::Mat& _keyPointMask);

            void computeDetectionHistograms();
            void train(const int &_start, const bool& _init);
            void deleteTracks();
            void associate();
            void update();
            void createNewClassifiers();
            void init();
            void checkOcclusions();
            bool overlapRoi(const int& idx_1, const int& idx_2, float& perc);
            bool overlapRoi(const cv::Rect& r1, const cv::Rect& _r2, float& perc);
    };

}
