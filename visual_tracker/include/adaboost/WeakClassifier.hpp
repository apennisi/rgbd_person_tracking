/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013-, Filippo Basso
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * WeakClassifier.hpp
 *
 *  Created on: Jul 15, 2011
 *      Author: Filippo Basso
 * 
 *  Modified by Andrea Pennisi
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#ifndef WEAKCLASSIFIER_H_
#define WEAKCLASSIFIER_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

namespace adaboost
{

class WeakClassifier
{

protected:
	int _lastPredictionCorrect;
	bool _selected;
	bool _new;
	int _marks;

public:
	WeakClassifier();
	virtual ~WeakClassifier();
	virtual int getSize() = 0;
	virtual void adjustSize(const cv::Rect & target) = 0;
	virtual void adjustSize(const cv::Size & target_size) = 0;

    virtual void train(const cv::Rect& target, int response, const std::pair<cv::Mat, cv::Mat>& integral) = 0;
    virtual void train(const cv::Point& location, int response, const std::pair<cv::Mat, cv::Mat>& integral) = 0;

    virtual int predict(const cv::Rect& target, const std::pair<cv::Mat, cv::Mat>& integral) = 0;
    virtual int predict(const cv::Point& location, const std::pair<cv::Mat, cv::Mat>& integral) = 0;

	virtual void getLimits(std::vector<std::pair<int, int> >& limits) = 0;

	bool islastPredictionCorrect() const;
	bool isNew() const;
	bool isSelected() const;
	void setSelected(bool b);
	void setNew(bool b);
	void markForRemoval();
	int getMarks() const;
	void reset();
};

inline WeakClassifier::WeakClassifier()
{
	_selected = false;
	_new = true;
	_marks = 0;
}

inline bool WeakClassifier::isNew() const
{
	return _new;
}

inline bool WeakClassifier::isSelected() const
{
	return _selected;
}

inline bool WeakClassifier::islastPredictionCorrect() const
{
	return _lastPredictionCorrect;
}

inline void WeakClassifier::markForRemoval()
{
	_marks++;
}

inline int WeakClassifier::getMarks() const
{
	return _marks;
}


inline void WeakClassifier::reset()
{
	_selected = false;
	_new = false;
	_marks = 0;
}

inline void WeakClassifier::setNew(bool b)
{
	_new = b;
}

inline void WeakClassifier::setSelected(bool b)
{
	_selected = b;
}

inline WeakClassifier::~WeakClassifier()
{
	// TODO Auto-generated constructor stub
}

/*******************************************************************************************************************/
template<class _T>
class WeakClassifier_ : public WeakClassifier
{
public:
    std::shared_ptr<_T> _feature;
    //_T *_feature;//TODO protected

	WeakClassifier_();
	WeakClassifier_(const _T & feature);
	WeakClassifier_(const WeakClassifier_ & copy);
	virtual ~WeakClassifier_();

	_T getFeature() const;
	void setFeature(const _T & feature);

	void getLimits(std::vector<std::pair<int, int> >& limits);
	int getSize();
	void adjustSize(const cv::Rect& target);
	void adjustSize(const cv::Size& target_size);

    void train(const cv::Rect& target, int response, const std::pair<cv::Mat, cv::Mat>& integral);
    virtual void train(const cv::Point& location, int response, const std::pair<cv::Mat, cv::Mat>& integral) = 0;

    int predict(const cv::Rect& target, const std::pair<cv::Mat, cv::Mat>& integral);
    virtual int predict(const cv::Point& location, const std::pair<cv::Mat, cv::Mat>& integral) = 0;

};

template<class _T>
inline WeakClassifier_<_T>::WeakClassifier_()
{
    _feature = std::shared_ptr<_T>(new _T());
    //_feature = new _T();
}

template<class _T>
inline WeakClassifier_<_T>::WeakClassifier_(const _T & feature)
{
	setFeature(feature);
}

template<class _T>
inline WeakClassifier_<_T>::WeakClassifier_(const WeakClassifier_ & copy)
{
	setFeature(*copy._feature);
}

template<class _T>
inline WeakClassifier_<_T>::~WeakClassifier_()
{
    //delete _feature;
}

template<class _T>
inline _T WeakClassifier_<_T>::getFeature() const
{
	return _feature;
}

template<class _T>
inline void WeakClassifier_<_T>::setFeature(const _T & feature)
{
    this->feature = std::shared_ptr<_T>(new _T(feature));
}

template<class _T>
inline void WeakClassifier_<_T>::getLimits(std::vector<std::pair<int, int> >& limits)
{
    this->_feature->getLimits(limits);
}

template<class _T>
int WeakClassifier_<_T>::getSize()
{
	std::vector<std::pair<int, int> > limits;
	getLimits(limits);
	int featureSize = 1;
	for (int i = 0; i < limits.size(); i++)
	{
		featureSize = featureSize * (limits[i].second-limits[i].first+1);
	}
	return featureSize;
}

template<class _T>
void WeakClassifier_<_T>::adjustSize(const cv::Rect & target)
{
	adjustSize(target.size());
}

template<class _T>
void WeakClassifier_<_T>::adjustSize(const cv::Size & target_size)
{
    _feature->resize(target_size);
}

template<class _T>
void WeakClassifier_<_T>::train(const cv::Rect &target, int response, const std::pair<cv::Mat, cv::Mat> &integral)
{
    adjustSize(target);
    return train(cv::Point(target.x, target.y), response, integral);
}

template<class _T>
int WeakClassifier_<_T>::predict(const cv::Rect &target, const std::pair<cv::Mat, cv::Mat> &integral)
{
    adjustSize(target);
    return predict(cv::Point(target.x, target.y), integral);
}

} /* namespace adaboost */

#endif /* WEAKCLASSIFIER_H_ */
