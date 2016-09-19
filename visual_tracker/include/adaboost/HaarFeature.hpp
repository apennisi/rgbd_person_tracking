/*
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2013-, Filippo Basso and Matteo Munaro
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
 * HaarFeature.hpp
 *
 *  Created on: Jul 14, 2011
 *      Author: Filippo Basso
 * 
 *  Modified by Andrea Pennisi
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#ifndef HAARFEATURE_H_
#define HAARFEATURE_H_

#include <opencv2/core/core.hpp>

namespace adaboost
{

template<int _N, int _C>
class HaarFeature_
{
protected:
	cv::Rect_<float> _rects[_N];
	cv::Rect _rects_r[_N];
	double _weights[_N];

	int _index;
	int _channel;

	double evaluateSingleRect(const cv::Point& location, const cv::Mat& integral, int index);

public:
	HaarFeature_();
	HaarFeature_(const HaarFeature_& copy);
	virtual ~HaarFeature_();

	void resize(const cv::Size& size);
	double evaluate(const cv::Point& location, const cv::Mat& integral);
	void addRect(const cv::Rect& target, const cv::Rect& rect, double weight);
};

typedef HaarFeature_<2, 3> HaarFeature2RGB;
typedef HaarFeature_<3, 3> HaarFeature3RGB;

//************************************************************************************//


template <int _N, int _C>
HaarFeature_<_N, _C>::HaarFeature_()
{
	_index = 0;
	_channel = rand() % _C;
}

template <int _N, int _C>
HaarFeature_<_N, _C>::HaarFeature_(const HaarFeature_<_N, _C>& copy)
{
	for(int index = 0; index < copy._index; index++)
	{
		_rects[index] = copy._rects[index];
		_rects_r[index] = copy._rects_r[index];
		_weights[index] = copy._weights[index];
	}
	_index = copy._index;
	_channel = copy._channel;
}

template <int _N, int _C>
HaarFeature_<_N, _C>::~HaarFeature_()
{
	// TODO Auto-generated destructor stub
}

template <int _N, int _C>
double HaarFeature_<_N, _C>::evaluateSingleRect(const cv::Point& location, const cv::Mat& integral, int index)
{
	cv::Rect* r = &_rects_r[index];

	if(location.y + r->y < 0 || location.x + r->x < 0 || location.y + r->y + r->height > integral.rows || location.x + r->x + r->width > integral.cols)
		return 0.0;

	double s = 0.0;

	s += integral.at<cv::Vec3d>(location.y + r->y + r->height, location.x + r->x + r->width)(_channel);
	s -= integral.at<cv::Vec3d>(location.y + r->y + r->height, location.x + r->x)(_channel);
	s -= integral.at<cv::Vec3d>(location.y + r->y, location.x + r->x + r->width)(_channel);
	s += integral.at<cv::Vec3d>(location.y + r->y, location.x + r->x)(_channel);
	return s;
}

template <int _N, int _C>
void HaarFeature_<_N, _C>::resize(const cv::Size& size)
{

	for(int index = 0; index < _N; index++)
	{
		cv::Rect_<float>* r = &_rects[index];
		_rects_r[index].x = lround(r->x * size.width);
		_rects_r[index].width = lround(r->width * size.width);
		_rects_r[index].y = lround(r->y * size.height);
		_rects_r[index].height = lround(r->height * size.height);
	}

}

template <int _N, int _C>
double HaarFeature_<_N, _C>::evaluate(const cv::Point& location, const cv::Mat& integral)
{
	double sum = 0.0;
	for(int index = 0; index < _N; index++)
	{
		sum += _weights[index] * evaluateSingleRect(location, integral, index);
	}
	return sum;
}

template <int _N, int _C>
void HaarFeature_<_N, _C>::addRect(const cv::Rect& target, const cv::Rect& rect, double weight)
{
	cv::Rect_<float> r;
	r.x = float(rect.x) / float(target.width);
	r.width = float(rect.width) / float(target.width);
	r.y = float(rect.y) / float(target.height);
	r.height = float(rect.height) / float(target.height);

	_rects_r[_index] = rect;
	_rects[_index] = r;
	_weights[_index] = weight;

	_index++;
}

} /* namespace adaboost */

#endif /* HAARFEATURE_H_ */
