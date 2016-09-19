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
 * KalmanWeakClassifier.hpp
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

#ifndef KALMANWEAKCLASSIFIER_H_
#define KALMANWEAKCLASSIFIER_H_

#include <adaboost/WeakClassifier.hpp>
#include <omp.h>

namespace adaboost
{

struct KalmanFilter
{
	float _P;
	float _K;
	float _mean;
	float _var;
	float _meanNoise;
	float _varNoise;
};

template <class _T>
class KalmanWeakClassifier : public WeakClassifier_<_T>
{
protected:
	adaboost::KalmanFilter _positive;
	adaboost::KalmanFilter _negative;

	float _theta;
	float _p;

	void init();

public:

	using WeakClassifier_<_T>::_feature;

	KalmanWeakClassifier();
	KalmanWeakClassifier(const _T& feature);
	KalmanWeakClassifier(const KalmanWeakClassifier& copy);
	virtual ~KalmanWeakClassifier();

    void train(const cv::Point& location, int response, const std::pair<cv::Mat, cv::Mat>& integral);
    int predict(const cv::Point& location, const std::pair<cv::Mat, cv::Mat>& integral);

};

template<class _T>
void KalmanWeakClassifier<_T>::init()
{
	_positive._P = _negative._P = 1000.0f;
	_positive._K = _negative._K = 0.0f;
	_positive._mean = _negative._mean = 0.0f;
	_positive._var = _negative._var = 0.0f;
	_positive._meanNoise = _negative._meanNoise = 0.0f;
	_positive._varNoise = _negative._varNoise = 0.01f;
}

template<class _T>
KalmanWeakClassifier<_T>::KalmanWeakClassifier() : WeakClassifier_<_T>()
{
	init();
}

template<class _T>
KalmanWeakClassifier<_T>::KalmanWeakClassifier(const _T& feature) : WeakClassifier_<_T>(feature)
{
	init();
}

template<class _T>
KalmanWeakClassifier<_T>::KalmanWeakClassifier(const KalmanWeakClassifier& copy) :  WeakClassifier_<_T>(copy)
{
	init();
	std::cout << "KalmanWeakClassifier<_T> copy constructor not implemented..." << std::endl;
}

template<class _T>
KalmanWeakClassifier<_T>::~KalmanWeakClassifier()
{
	// TODO Auto-generated destructor stub
}

template<class _T>
void KalmanWeakClassifier<_T>::train(const cv::Point& location, int response, const std::pair<cv::Mat, cv::Mat>& integral)
{
    KalmanFilter* filter;
    float measure = float(_feature->evaluate(integral));

    filter = (response == 1) ? &_positive : &_negative;

    filter->_K = filter->_P / (filter->_P + filter->_varNoise);
    filter->_mean = filter->_K * measure + (1 - filter->_K) * filter->_mean;
    filter->_var = filter->_K * pow(measure - filter->_mean, 2) + (1 - filter->_K) * filter->_var;
    filter->_P = (1 - filter->_K) * filter->_P;

    _p = copysign(1.0f, _positive._mean - _negative._mean);
    _theta = (_positive._mean + _negative._mean) * 0.5f;

    this->_lastPredictionCorrect = (lround(_p * copysign(1.0f, measure - _theta)) == response);

}

template<class _T>
int KalmanWeakClassifier<_T>::predict(const cv::Point &location, const std::pair<cv::Mat, cv::Mat> &integral)
{

    float measure = float(this->_feature->evaluate(integral));
    return lround(_p * copysign(1.0f, measure - _theta));
}

} /* namespace adaboost */

#endif /* KALMANWEAKCLASSIFIER_H_ */
