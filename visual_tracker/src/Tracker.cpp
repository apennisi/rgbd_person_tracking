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


#include "tracker/Tracker.h"
#include <ctime>
#include <iterator>
#include <stack>
#include <ros/ros.h>

using namespace tracker;

Tracker::Tracker(const int &_classifiers, const int &_selectors, const int &_numFeatureToDraw, const float &_assocThres, const colorspace &_color,
                 const int& _accDetection, const int& _lossDetection)
    : classifiers_(_classifiers), selectors_(_selectors), numFeaturesToDraw_(_numFeatureToDraw), assocThres_(_assocThres), color_(_color),
      accDetection_(_accDetection), lossDetection_(_lossDetection)
{
    initialize_ = true;
    idCounter_ = 0;
    cZeroDet_ = 0;
    rng = cv::RNG(12345);
}

int counter = 0;
void Tracker::track(const std::vector<detection_struct> &_detections, const cv::Mat &_image)
{
    negMasks_.clear();
    newTracks.clear();
    histograms_.clear();
    featureHistograms_.clear();
    keypoints_.clear();

    if(_detections.size() == 0)
    {
        cZeroDet_++;
        if(cZeroDet_ == 5)
        {
            tracks_.clear();
            strongClassifiers_.clear();
            detections_.clear();
            initialize_ = true;
        }
        return;
    }

    cZeroDet_ = 0;

    detections_ = _detections;

    cv::Ptr<cv::FeatureDetector> detector = new cv::FastFeatureDetector;
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor;
    detector->detect(_image, keypoints_);
    extractor->compute(_image, keypoints_, descriptors_);

    keyPointImage_ = cv::Mat(_image.size(), CV_32FC1, cv::Scalar(-1));
    keyPointMask_ = cv::Mat(_image.size(), CV_8UC1, cv::Scalar(0));

    int i = 0;
    for(const auto &key : keypoints_)
    {
        keyPointImage_.at<float>(key.pt) = i;
        keyPointMask_.at<uchar>(key.pt) = uchar(255);
        ++i;
    }
    switch (color_) {
    case HSV:
        cv::cvtColor(_image, image_, CV_BGR2HSV);
        break;
    case LUV:
        cv::cvtColor(_image, image_, CV_BGR2Luv);
        break;
    case LAB:
        cv::cvtColor(_image, image_, CV_BGR2Lab);
    default:
        image_ = _image.clone();
        break;
    }
    if(initialize_)
    {
        /*1*/init(); //Initialize the classifier
        /*2*/computeDetectionHistograms(); //Compute the histrograms and the masks
        /*3*/train(0, true); //Train the classifier
        initialize_ = false;
    }
    else
    {
        /*1*/computeDetectionHistograms(); //Compute the histrograms and the masks
        /*2*/associate(); //Associate the detections with the classifiers
        /*4*/checkOcclusions();//Check occluded tracks
        /*3*/update(); //Update old classifiers
        /*5*/deleteTracks(); //Delete unused tracks
        /*6*/createNewClassifiers(); //Create the classifiers for the new tracks
        int start = strongClassifiers_.size() - newTracks.size();
        /*7*/train(start, false); //Train the new classifiers
    }

    counter++;
}

void Tracker::calcColorHistogram(const cv::Mat &_image, const cv::Mat &_blobMask, const int &_bins, const int &_channels, cv::Mat &_histogram)
{
    const int histSize[] = {_bins, _bins, _bins};
    static const float range[] = {0, 255};
    static const int ch[] = {0, 1, 2};
    static const float* histRanges[] = {range, range, range};

    cv::Mat imageToHist;    // image used to compute the color histogram
    _image.copyTo(imageToHist, _blobMask);	// the image is filtered with a mask

    cv::calcHist(&imageToHist, 1, ch, _blobMask, _histogram, _channels, histSize, histRanges, true, false);   // color histogram computation
    _histogram /= cv::countNonZero(_blobMask);   // histogram normalization
}


void Tracker::createNegativeHistograms(const cv::Mat &_image, const cv::Mat &_mask, const int &_n, const int &_bins, std::vector<cv::Mat> &_negative_histograms,
                                        std::vector<cv::Mat>& _negative_feature_histograms, const std::vector<detection_struct> &_detections, const int &_det)
{
    // creates histograms of random image patches to be used as negative examples
    cv::Mat histogram;
    cv::Mat featureHistogram;
    cv::Mat neg_image;
    cv::Mat neg_mask;
    bool is_contained;
    int x_min;
    int y_min;
    int width;
    int height;
    int min_height = 10;
    int min_width = 10;
    cv::Rect neg_rect;
    std::thread hist, feat;

    for(int i = 0; i < _n; ++i)
    {

        is_contained = true;

        while(is_contained)
        {
            x_min = rand() % (_image.cols - min_width - 1);
            y_min = rand() % (_image.rows - min_height - 1);
            width = rand() % (_image.cols - x_min - 1);
            height = rand() % (_image.rows - y_min - 1);
            is_contained = cv::countNonZero(_mask(cv::Rect(x_min, y_min, width, height))) == 0;
        }


        neg_rect = cv::Rect(x_min, y_min, width, height);

        neg_image = _image(neg_rect);
        neg_mask = _mask(neg_rect);


        feat = std::thread(std::bind(calcFeatures, neg_rect, std::ref(featureHistogram), descriptors_, keyPointImage_, keyPointMask_));
        hist = std::thread(std::bind(calcColorHistogram, neg_image, neg_mask, _bins, _image.channels(),  std::ref(histogram)));
        hist.join();
        feat.join();

        _negative_histograms.push_back(histogram.clone());
        _negative_feature_histograms.push_back(featureHistogram.clone());
    }

    for(int i = 0; i < _detections.size(); ++i)
    {
        if(i != _det)
        {
            _negative_histograms.push_back(histograms_[i]);
            _negative_feature_histograms.push_back(featureHistograms_[i]);
        }
    }
}

void Tracker::calcFeatures(const cv::Rect& _detection, cv::Mat& _histrogram, const cv::Mat& _descriptors,
                           const cv::Mat& _keyPointImage, const cv::Mat& _keyPointMask)
{
    const int histSize[] = {64};
    static const float range[] = {0, 255};
    static const float* histRange[] = {range};
    static const int ch[1] = {0};

    cv::Mat mask = _keyPointMask(_detection);
    cv::Mat feat = _keyPointImage(_detection);

    cv::Mat nonZeroCoordinates;

    cv::findNonZero(mask, nonZeroCoordinates);
    cv::Mat descriptor = cv::Mat(cv::Size(64, nonZeroCoordinates.total()), CV_32FC1);

    float idx;

    for(uint j = 0; j < nonZeroCoordinates.total(); ++j)
    {
        idx = feat.at<float>(nonZeroCoordinates.at<cv::Point>(j));
        _descriptors.row(idx).copyTo(descriptor.row(j));
    }

    cv::Mat blobMask = cv::Mat(descriptor.size(), CV_8UC1, cv::Scalar(255));
    cv::calcHist(&descriptor, 1, ch, blobMask, _histrogram, 1, histSize, histRange, true, false);
    _histrogram /= int(descriptor.cols);
}


void Tracker::computeDetectionHistograms()
{
    cv::Mat negativeMask;
    cv::Mat target_image;
    cv::Mat blobMask;
    cv::Mat histogram;
    cv::Mat featureHist;

    cv::Rect detection;
    std::thread hist, feat;

    for(uint i = 0; i < detections_.size(); ++i)
    {
        detection = detections_.at(i).bbox;
        if(detection.x + detection.width > image_.cols)
        {
            detection.width = image_.cols - detection.x  - 1;
        }

        if(detection.x < 0)
        {
            detection.x = 0;
        }

        if(detection.y + detection.height > image_.rows)
        {
            detection.height = image_.rows - detection.y - 1;
        }

        if(detection.y < 0)
        {
            detection.y = 0;
        }

        target_image = image_(detection);
        blobMask = cv::Mat(target_image.rows, target_image.cols, CV_8UC1, cv::Scalar(255));
        negativeMask = cv::Mat(image_.rows, image_.cols, CV_8UC1, cv::Scalar(255));
        negativeMask(detection) = cv::Scalar(0);
        hist = std::thread(std::bind(calcColorHistogram, target_image, blobMask, BINS, CHANNELS,  std::ref(histogram)));
        feat = std::thread(std::bind(calcFeatures, detection, std::ref(featureHist), descriptors_, keyPointImage_, keyPointMask_));
        hist.join();
        feat.join();
        negMasks_.push_back(negativeMask.clone());
        histograms_.push_back(histogram.clone());
        featureHistograms_.push_back(featureHist.clone());
    }
}

void Tracker::train(const int &_start, const bool& _init)
{
    std::vector<cv::Mat> hists;
    std::vector<cv::Mat> featHists;
    std::vector<cv::Mat> negMasks;

    if(_init)
    {
        //ROS_ERROR("QUI");
        hists = histograms_;
        featHists = featureHistograms_;
        negMasks = negMasks_;
    }
    else
    {
        int newTrack;
        //#pragma omp parallel for private(newTrack)
        for(uint i = 0; i < newTracks.size(); ++i)
        {
            newTrack = newTracks.at(i);
            hists.push_back(histograms_.at(newTrack));
            featHists.push_back(featureHistograms_.at(newTrack));
            negMasks.push_back(negMasks_.at(newTrack));
        }
    }

    std::thread positiveSamples, tracks;
    positiveSamples = std::thread([this](const int& _start, const std::vector<cv::Mat>& hists, const std::vector<cv::Mat>& featHists, const std::vector<cv::Mat>& negMasks)
    {
        int counter = _start;
        int selection;


        for(uint i = 0; i < hists.size(); ++i)
        {
            selection = (newTracks.size() == 0) ? i : newTracks.at(i);
            strongClassifiers_[counter]->update(cv::Point(0, 0), 1, std::make_pair(hists.at(i), featHists.at(i)), numFeaturesToDraw_);
            strongClassifiers_[counter]->replaceWorstWeakClassifier<_ColorWeakClassifier>();

            std::vector<cv::Mat> wrongHistograms;
            std::vector<cv::Mat> wrongFeatureHistrograms;
            createNegativeHistograms(image_, negMasks[i], 10, BINS, wrongHistograms, wrongFeatureHistrograms, detections_, selection);

            // Update with negative examples:
            for(int k = 0; k < wrongHistograms.size(); k++)
            {
                strongClassifiers_[counter]->update(cv::Point(0, 0), -1, std::make_pair(wrongHistograms[k], wrongFeatureHistrograms[k]), numFeaturesToDraw_);
                strongClassifiers_[counter]->replaceWorstWeakClassifier<_ColorWeakClassifier>();
            }

            ++counter;
        }

    }, _start, hists, featHists, negMasks);

    tracks = std::thread([this](const uint& _n, const std::vector<cv::Mat>& hists, const std::vector<cv::Mat>& featHists)
    {
           cv::Scalar color;
           cv::Rect detection;
           for(uint i = 0; i < _n; ++i)
           {
               detection = detections_.at( (newTracks.size() == 0) ? i : newTracks.at(i)  ).bbox;
               color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
               std::shared_ptr<Track> tr(new Track(detection.tl(), detection.width, detection.height, color));

               tr->setWidth( detection.width );

               tr->setHeight( detection.height );

               tr->setHist(hists.at(i).clone());
               tr->setFeatHist(featHists.at(i).clone());
               tracks_.push_back(tr);
           }
    }, hists.size(), hists, featHists);
    positiveSamples.join();
    tracks.join();


    for(const auto& track : tracks_)
    {
        track->getPosition();
    }

    for (int i =  tracks_.size() - 1; i > 0;  --i)
    {
        cv::Mat p = tracks_[i]->lastPosition();
        cv::Rect r1(p.at<float>(0), p.at<float>(1), p.at<float>(4), p.at<float>(5));
        float perc_1, perc_2;
        for (auto j = i-1; j >= 0; j--)
        {
            cv::Mat p2 = tracks_[j]->lastPosition();
            cv::Rect r2(p2.at<float>(0), p2.at<float>(1), p2.at<float>(4), p2.at<float>(5));
            if (overlapRoi(r1, r2, perc_1))
            {
                overlapRoi(r2, r1, perc_2);
                if(perc_1 > 0.98 || perc_2 > 0.98)
                {
                    if(tracks_[j]->numDetections() < tracks_[i]->numDetections())
                    {
                        tracks_[j] = tracks_[i];
                        strongClassifiers_[j] = strongClassifiers_[i];
                    }
                    tracks_.erase(tracks_.begin() + i);
                    strongClassifiers_.erase(strongClassifiers_.begin() + i);
                    break;
                }
            }
        }
    }


}

void Tracker::deleteTracks()
{
    for(const auto &cls : classifiersToDelete)
    {
        tracks_.erase(tracks_.begin() + cls);
        strongClassifiers_.erase(strongClassifiers_.begin() + cls);
    }
}

void Tracker::associate()
{
    oldTracks.clear();
    newTracks.clear();
    classifiersToDelete.clear();

    int N = tracks_.size();
    int M = detections_.size();

    cv::Mat classifierCosts(cv::Size(M, N), CV_64FC1);
    cv::Mat distanceCosts(cv::Size(M, N), CV_64FC1);

    std::vector<std::vector<float> > predictionMap;
    float prediction;
    std::vector<float> predictions;

    double dist;
    cv::Point2d diff;
    //COST COMPUTATION
    for(uint i=0; i < tracks_.size(); ++i)
    {
        for(uint j=0; j < detections_.size(); ++j)
        {
            diff = tracks_[i]->lastDetection() - cv::Point(detections_[j].bbox.x + detections_[j].bbox.width/2, detections_[j].bbox.y + detections_[j].bbox.height/2);
            dist = sqrtf(diff.x*diff.x + diff.y*diff.y);
            distanceCosts.at<double>(i, j) = dist;
        }
    }

    for(uint k = 0; k < histograms_.size(); ++k)
    {
        predictions.clear();
        for(int j = 0; j < strongClassifiers_.size(); ++j)
        {
            prediction = strongClassifiers_[j]->predict(cv::Point(0, 0), std::make_pair(histograms_[k], featureHistograms_[k]));
            classifierCosts.at<double>(j, k) = strongClassifiers_[j]->predict(cv::Point(0, 0), std::make_pair(histograms_[k], featureHistograms_[k]));
            predictions.push_back(prediction);
        }
        predictionMap.push_back(predictions);
    }

    cv::Mat x(distanceCosts.size(), distanceCosts.type(), cv::Scalar(-1.));

    for(uint i = 0; i < distanceCosts.rows; ++i)
    {
        for(uint j = 0; j < distanceCosts.cols; ++j)
        {
            if(distanceCosts.at<double>(i, j) < 20.)
            {
                x.at<double>(i, j) = 0.;
            }
        }
    }

    for(uint i = 0; i < x.rows; ++i)
    {
        for(uint j = 0; j < x.cols; ++j)
        {
            if(x.at<double>(i, j) == 0.)
            {
                for(uint k = i + 1; k < x.rows; ++k)
                {
                    if(x.at<double>(k, j) == 0.)
                    {
                        if(tracks_[i]->numDetections() > tracks_[k]->numDetections())
                        {
                            x.at<double>(k, j) = -1.;
                        }
                        else
                        {
                            x.at<double>(i, j) = -1.;
                            break;
                        }
                    }
                }
            }
        }
    }

    //PREDICTION ANALYSIS AND ASSOCIATION
    std::vector<bool> usedClassifiers(tracks_.size(), false);
    int i = 0;

    for(const auto &pr : predictionMap)
    {
        const auto &m = std::max_element(pr.begin(), pr.end());
        if(*m > 0.)
        {
            uint idx = (int)(m - pr.begin());
            if(oldTracks.find(idx) != oldTracks.end())
            {
                std::pair<int, float> elem = oldTracks[idx];
                if(elem.second < *m)
                {
                    elem = std::make_pair(i, *m);
                }
                x.at<double>(idx, i) = -1;
            }
            else
            {
                oldTracks.insert(std::make_pair(idx, std::make_pair(i, *m)));
                x.at<double>(idx, i) = -1;
            }
            usedClassifiers.at(idx) = true;
        }
        else
        {
            newTracks.push_back(i);
        }
        ++i;
    }

#if 1
    if(counter != 1)
    {
        for(uint i = 0; i < x.rows; ++i)
        {
            for(uint j = 0; j < x.cols; ++j)
            {
                if(x.at<double>(i, j) == 0.)
                {
                    const auto el = std::find(newTracks.begin(), newTracks.end(), j);
                    if(el != newTracks.end())
                    {
                        newTracks.erase(el);
                    }
                    if(oldTracks.find(i) != oldTracks.end())
                    {
                        std::pair<int, float> elem = oldTracks[i];
                        if(elem.second < classifierCosts.at<double>(i, j))
                        {
                            elem = std::make_pair(j, classifierCosts.at<double>(i, j));
                        }
                    }
                    else
                    {
                        oldTracks.insert(std::make_pair(i, std::make_pair(j, classifierCosts.at<double>(i, j))));
                    }
                    usedClassifiers.at(i) = true;
                }

            }
        }
    }
#endif

    //SELECT WHICH CLASSIFIERS HAS TO BE DELETED
    i = 0;

    for(const auto &usedClassifier : usedClassifiers)
    {
        if(!usedClassifier)
        {
            if(tracks_[i]->lossDetections() == lossDetection_)
            {
                classifiersToDelete.push_back(i);
            }
            else
            {
                tracks_[i]->addLossDetection();
            }
        }
        ++i;
    }
}

void Tracker::update()
{
#if 0
    #pragma omp parallel for num_threads(omp_get_num_procs() * omp_get_num_threads())
    for(const auto &track : oldTracks)
    {
        ///Update Positive sample
        strongClassifiers_[track.first]->update(cv::Point(0, 0), 1, std::make_pair(histograms_.at(track.second.first), featureHistograms_.at(track.second.first)), numFeaturesToDraw_);
        strongClassifiers_[track.first]->replaceWorstWeakClassifier<_ColorWeakClassifier>();

        ///Update Negative sample
        // Create negative examples (randomly chosen from the rest of the image + the other detections):
        #pragma omp for nowait
        {
            std::vector<cv::Mat> wrongHistograms;
            std::vector<cv::Mat> wrongFeatureHistrograms;
            createNegativeHistograms(image_, negMasks_[track.second.first], 10, BINS, wrongHistograms, wrongFeatureHistrograms, detections_, track.second.first);

            // Update with negative examples:
            for(int k = 0; k < wrongHistograms.size(); k++)
            {
                strongClassifiers_[track.first]->update(cv::Point(0, 0), -1,std::make_pair(wrongHistograms[k], wrongFeatureHistrograms[k]), numFeaturesToDraw_);
                strongClassifiers_[track.first]->replaceWorstWeakClassifier<_ColorWeakClassifier>();
            }
        }

        #pragma omp for nowait
        {
            tracks_[track.first]->getPosition();
            tracks_[track.first]->update(detections_.at(track.second.first).tl(), detections_.at(track.second.first).width, detections_.at(track.second.first).height);
            tracks_[track.first]->setWidth(detections_.at(track.second.first).width);
            tracks_[track.first]->setHeight(detections_.at(track.second.first).height);
            tracks_[track.first]->addDetection();
            tracks_[track.first]->setLastDetection(detections_.at(track.second.first).tl());
            tracks_[track.first]->setHist(histograms_.at(track.second.first));
            tracks_[track.first]->setFeatHist(featureHistograms_.at(track.second.first));
            if(tracks_[track.first]->numDetections() >= accDetection_ && tracks_[track.first]->getId() == -1)
            {
                tracks_[track.first]->setId(idCounter_++);
            }
        }
    }
#else
    std::thread tracks, positiveSamples;

    positiveSamples = std::thread([this]()
    {
        auto track = oldTracks.begin();
        for(uint i = 0; i < oldTracks.size(); ++i)
        {
            ///Update Positive sample
            strongClassifiers_[track->first]->update(cv::Point(0, 0), 1, std::make_pair(histograms_.at(track->second.first), featureHistograms_.at(track->second.first)), numFeaturesToDraw_);
            strongClassifiers_[track->first]->replaceWorstWeakClassifier<_ColorWeakClassifier>();

            ///Update Negative sample
            // Create negative examples (randomly chosen from the rest of the image + the other detections):
            std::vector<cv::Mat> wrongHistograms;
            std::vector<cv::Mat> wrongFeatureHistrograms;
            createNegativeHistograms(image_, negMasks_[track->second.first], 10, BINS, wrongHistograms, wrongFeatureHistrograms, detections_, track->second.first);
            // Update with negative examples:
            for(int k = 0; k < wrongHistograms.size(); k++)
            {
                strongClassifiers_[track->first]->update(cv::Point(0, 0), -1,std::make_pair(wrongHistograms[k], wrongFeatureHistrograms[k]), numFeaturesToDraw_);
                strongClassifiers_[track->first]->replaceWorstWeakClassifier<_ColorWeakClassifier>();
            }
            ++track;
        }
    });


    tracks = std::thread([this]()
    {
        for(const auto &track : oldTracks)
        {
            tracks_[track.first]->getPosition();
            tracks_[track.first]->update(detections_.at(track.second.first).bbox.tl(), detections_.at(track.second.first).bbox.width, detections_.at(track.second.first).bbox.height);
            tracks_[track.first]->setWidth(detections_.at(track.second.first).bbox.width);
            tracks_[track.first]->setHeight(detections_.at(track.second.first).bbox.height);
            tracks_[track.first]->addDetection();
            tracks_[track.first]->setLastDetection(detections_.at(track.second.first).bbox.tl());
            tracks_[track.first]->setHist(histograms_.at(track.second.first));
            tracks_[track.first]->setFeatHist(featureHistograms_.at(track.second.first));
            tracks_[track.first]->set3DPoint(detections_.at(track.second.first).point3D);
            if(tracks_[track.first]->numDetections() >= accDetection_ && tracks_[track.first]->getId() == -1)
            {
                tracks_[track.first]->setId(idCounter_++);
            }
        }
    });

    tracks.join();
    positiveSamples.join();

#endif
}

void Tracker::createNewClassifiers()
{
    for(int i = 0; i < newTracks.size(); ++i)
    {
        std::shared_ptr<adaboost::StrongClassifier> classifier(new adaboost::StrongClassifier(classifiers_, selectors_));
        for(int j = 0; j < classifiers_; j++)
            classifier->createWeakClassifier<_ColorWeakClassifier>();
        strongClassifiers_.push_back(classifier);
    }
}

void Tracker::init()
{
    strongClassifiers_.resize(detections_.size());
    for(int i = 0; i < detections_.size(); ++i)
    {
        std::shared_ptr<adaboost::StrongClassifier> classifier(new adaboost::StrongClassifier(classifiers_, selectors_));
        for(int j = 0; j < classifiers_; j++)
            classifier->createWeakClassifier<_ColorWeakClassifier>();
        strongClassifiers_[i] = classifier;
    }
}

void Tracker::checkOcclusions()
{


    float percOcc;
    float percOcc2;

    auto track = oldTracks.begin();

    for(uint i = 0; i < oldTracks.size(); ++i)
    {
        for(const auto& oTrack : oldTracks)
        {
            if(track->second.first != oTrack.second.first)
            {
                if(overlapRoi(track->second.first, oTrack.second.first, percOcc))
                {
                    if(percOcc >= 0.4)
                    {
                        histograms_[track->second.first] = tracks_[track->first]->hist().clone();
                        histograms_[oTrack.second.first] = tracks_[oTrack.first]->hist().clone();
                        featureHistograms_[track->second.first] = tracks_[track->first]->featHist().clone();
                        featureHistograms_[oTrack.second.first] = tracks_[oTrack.first]->featHist().clone();
                    }
                }
            }
        }
        ++track;
    }
}

bool Tracker::overlapRoi(const int& idx_1, const int& idx_2, float& perc)
{
    cv::Rect rect_1 = detections_[idx_1].bbox;
    cv::Rect rect_2 = detections_[idx_2].bbox;
    int x_tl = fmax(rect_1.x, rect_2.x);
    int y_tl = fmax(rect_1.y, rect_2.y);
    int x_br = fmin(rect_1.x + rect_1.width, rect_2.x + rect_2.width);
    int y_br = fmin(rect_1.y + rect_1.height, rect_2.y + rect_2.height);
    int w, h;
    if (x_tl < x_br && y_tl < y_br)
    {
        w = x_br - x_tl;
        h = y_br - y_tl;
        perc = (w*h) / float(rect_1.area());
        return true;
    }
    perc = 0;
    return false;
}

bool Tracker::overlapRoi(const cv::Rect &r1, const cv::Rect &_r2, float &perc)
{
    int x_tl = fmax(r1.x, _r2.x);
    int y_tl = fmax(r1.y, _r2.y);
    int x_br = fmin(r1.x + r1.width, _r2.x + _r2.width);
    int y_br = fmin(r1.y + r1.height, _r2.y + _r2.height);
    int w, h;
    if (x_tl < x_br && y_tl < y_br)
    {
        w = x_br - x_tl;
        h = y_br - y_tl;
        perc = (w*h) / float(r1.area());
        return true;
    }
    perc = 0;
    return false;
}

void Tracker::visualize(cv::Mat &_image, const double& scaling_factor)
{
    std::stringstream prediction_ss;
    cv::Mat p;
    for(const auto &track : tracks_)
    {
        if(track->numDetections() >= accDetection_ && !track->hide())
        {
            prediction_ss.str("");
            prediction_ss << track->getId();
            p = track->lastPosition() * scaling_factor;
            cv::rectangle(_image, cv::Rect(p.at<float>(0), p.at<float>(1), p.at<float>(4), p.at<float>(5)), track->getColor(), 2);
            cv::putText(_image, prediction_ss.str(), cv::Point(p.at<float>(0), p.at<float>(1)), cv::FONT_HERSHEY_SIMPLEX,
                        0.55, cv::Scalar(0, 255, 0), 2, CV_AA);
        }
    }
}

void Tracker::generateMessage(people_msgs::Tracks &_msg, const double& scaling_factor)
{
    cv::Mat p;
    people_msgs::Track t;
    for(const auto &track : tracks_)
    {
        if(track->numDetections() >= accDetection_ && !track->hide())
        {
            const cv::Point3d& point3D = track->get3DPoint();
            p = track->lastPosition() * scaling_factor;
            cv::Rect r = cv::Rect(p.at<float>(0), p.at<float>(1), p.at<float>(4), p.at<float>(5));
            t.id = track->getId();
            t.point3D.x = point3D.x;
            t.point3D.y = point3D.y;
            t.point3D.z = point3D.z;
            t.point2D.x = r.x + r.width >> 1;
            t.point2D.y = r.y + r.height;
            _msg.tracks.push_back(t);
        }
    }
}

Tracker::~Tracker()
{
}
