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
 * Selector.cpp
 *
 *  Created on: Jul 17, 2011
 *      Author: Filippo Basso
 * 
 *  Modified by Andrea Pennisi
 *
 * If you use part of this code, please cite:
 *    F. Basso, M. Munaro, S. Michieletto and E. Menegatti. Fast and robust multi-people tracking from RGB-D data for a mobile robot.
 *    In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.
 */

#include <adaboost/Selector.hpp>
#include <cmath>
#include <omp.h>

namespace adaboost
{

Selector::Selector(int size)
{
	for(int i = 0; i < size; i++)
	{
		_weights.push_back(new Weights);
	}
	_votingWeight = 0.0f;
    _selectedWeak = 0;
}

Selector::~Selector()
{
	for(uint i = 0; i < _weights.size(); i++)
	{
		delete _weights[i];
	}
}

std::shared_ptr<WeakClassifier> Selector::selectBest(const std::vector<std::shared_ptr<WeakClassifier> >& classifiers, float& lambda)
{
	float minError = 1.0f;

    std::shared_ptr<WeakClassifier> _worstWeak;
	float maxError = 0.0f;
    float error;
    Weights* w;
    std::shared_ptr<WeakClassifier> wc;
    uint end = _weights.size();

    for(uint i = 0; i < end; i++)
	{
        wc = classifiers[i];
        w = _weights[i];
		if(wc->isNew())
		{
			w->corr = 1.0f;
			w->wrong = 1.0f;
		}
        if(wc->islastPredictionCorrect())
			w->corr += lambda;
		else
			w->wrong += lambda;

        if(!wc->isSelected() )
		{
            error = w->wrong / (w->wrong + w->corr);

            if(error < minError)
            {
                minError = error;
                _selectedWeak = classifiers[i];
            }
            else if(error >= maxError)
            {
				maxError = error;
                _worstWeak = classifiers[i];
            }
		}
    }

	_worstWeak->markForRemoval();
	_selectedWeak->setSelected(true);

	if(_selectedWeak->islastPredictionCorrect())
		lambda /= (2 * (1.0f - minError));
	else
		lambda /= (2 * minError);

	_votingWeight = 0.5f * std::log((1.0f - minError) / minError);

	return _selectedWeak;
}

float Selector::predict(const cv::Rect& target, const std::pair<cv::Mat, cv::Mat>& integral)
{
	return _votingWeight * _selectedWeak->predict(target, integral); //TODO this method resizes the feature
}

float Selector::predict(const cv::Point& location, const std::pair<cv::Mat, cv::Mat>& integral)
{
    return _votingWeight * _selectedWeak->predict(location, integral); //TODO this method resizes the feature
}

float Selector::getWeight()
{
	return _votingWeight;
}
} /* namespace adaboost */
