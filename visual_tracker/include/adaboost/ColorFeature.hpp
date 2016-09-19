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
 * ColorFeature.hpp
 *
 *  Created on: Aug 31, 2011
 *      Author: Filippo Basso
 * 
 *  Modified by Andrea Pennisi
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#ifndef COLORHIST_H_
#define COLORHIST_H_

#include <opencv2/core/core.hpp>
#include <utility>

namespace adaboost
{

template<int _N, int _C>
class ColorFeature_
{
public:
	ColorFeature_();
	ColorFeature_(const ColorFeature_<_N, _C>& copy);
	virtual ~ColorFeature_();

	void resize(const cv::Size& size);
	double evaluate(const cv::Point& location, const cv::Mat& data);

	void getLimits(std::vector<std::pair<int, int> >& limits);

};

template<int _N>
class ColorFeature_<_N, 1>
{
protected:
	int i1, i2;

public:
	ColorFeature_();
	ColorFeature_(const ColorFeature_<_N, 1>& copy);
	virtual ~ColorFeature_();

	void resize(const cv::Size& size);
	double evaluate(const cv::Point& location, const cv::Mat& data);

	void getLimits(std::vector<std::pair<int, int> >& limits);
};

template<int _N>
class ColorFeature_<_N, 3>
{
protected:
	int i1, i2, j1, j2, k1, k2;

public:
	ColorFeature_();
	ColorFeature_(const ColorFeature_<_N, 3>& copy);
	virtual ~ColorFeature_();

	void resize(const cv::Size& size);
	double evaluate(const cv::Point& location, const cv::Mat& data);

	void getLimits(std::vector<std::pair<int, int> >& limits);
};

typedef ColorFeature_<16, 3> ColorFeature16RGB;

//************************************************************************************//

template <int _N>
ColorFeature_<_N, 1>::ColorFeature_()
{
	i2 = rand() % (_N) + 1;
	i1 = rand() % i2;
}

template <int _N>
ColorFeature_<_N, 3>::ColorFeature_()
{
	i2 = rand() % (_N) + 1;
	i1 = rand() % i2;

	j2 = rand() % (_N) + 1;
	j1 = rand() % j2;

	k2 = rand() % (_N) + 1;
	k1 = rand() % k2;

}

template <int _N>
ColorFeature_<_N, 1>::ColorFeature_(const ColorFeature_<_N, 1>& copy)
{
	i1 = copy.i1;
	i2 = copy.i2;
}

template <int _N>
ColorFeature_<_N, 3>::ColorFeature_(const ColorFeature_<_N, 3>& copy)
{
	i1 = copy.i1;
	i2 = copy.i2;
	j1 = copy.j1;
	j2 = copy.j2;
	k1 = copy.k1;
	k2 = copy.k2;
}

template <int _N>
ColorFeature_<_N, 1>::~ColorFeature_()
{
	// TODO Auto-generated destructor stub
}

template <int _N>
ColorFeature_<_N, 3>::~ColorFeature_()
{
	// TODO Auto-generated destructor stub
}

template <int _N>
void ColorFeature_<_N, 1>::getLimits(std::vector<std::pair<int, int> >& limits)
{
    limits.push_back(std::make_pair(i1, i2));
}

template <int _N>
void ColorFeature_<_N, 3>::getLimits(std::vector<std::pair<int, int> >& limits)
{
    limits.push_back(std::make_pair(i1, i2));
    limits.push_back(std::make_pair(j1, j2));
    limits.push_back(std::make_pair(k1, k2));
}

template <int _N>
void ColorFeature_<_N, 1>::resize(const cv::Size& size)
{
	//Nothing
}

template <int _N>
void ColorFeature_<_N, 3>::resize(const cv::Size& size)
{
	//Nothing
}

template <int _N>
double ColorFeature_<_N, 3>::evaluate(const cv::Point& location, const cv::Mat& data)
{

	double sum = 0.0;
	for(int i = i1; i < i2; i++)
		for(int j = j1; j < j2; j++)
			for(int k = k1; k < k2; k++)
				sum += data.at<float>(i, j, k);

	//ROS_INFO("%f", sum);

	return sum;
}

template <int _N>
double ColorFeature_<_N, 1>::evaluate(const cv::Point& location, const cv::Mat& data)
{

	double sum = 0.0;
	for(int i = i1; i < i2; i++)
		sum += data.at<float>(i);

	//ROS_INFO("%f", sum);

	return sum;
}

} /* namespace adaboost */


#endif /* COLORHIST_H_ */
