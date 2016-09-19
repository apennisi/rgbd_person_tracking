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
 * StrongClassifier.cpp
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

#include <adaboost/StrongClassifier.hpp>
#include <ctime>
#include <stack>

namespace adaboost
{

StrongClassifier::StrongClassifier()
{
	// TODO Auto-generated constructor stub
}

StrongClassifier::StrongClassifier(int size, int selectors)
{
    _weakClassifiers.reserve(size);
    _selectors.reserve(selectors);
	for(int i = 0; i < selectors; i++)
	{
//		_selectors.push_back(new Selector(size));
        _selectors.push_back(std::shared_ptr<Selector>(new Selector(size)));
	}
}

StrongClassifier::~StrongClassifier()
{
}

void StrongClassifier::updateTarget(const cv::Rect& rect)
{
    updateTarget(rect.size());
}

std::stack<clock_t> tictoc_stack;
void tic() {
    tictoc_stack.push(clock());
}

void toc(const std::string& s) {
    std::cout << s << "- Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}

void StrongClassifier::update(const cv::Point &location, int response, const std::pair<cv::Mat, cv::Mat> &integral, int numFeaturesToPublish)
{

	for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
		_weakClassifiers[i]->train(location, response, integral);
	}


    if(_selectors.size() > 0)
    {
        _bestFeatures.clear();
    }


    std::shared_ptr<Selector> s;
    std::shared_ptr<adaboost::WeakClassifier> b;
    std::vector<std::pair<int, int> > v;
    float lambda = 1.0f;

    for(size_t j = 0; j < _selectors.size(); j++)
    {
        s = _selectors[j];

        b = s->selectBest(_weakClassifiers, lambda);

        if(j < numFeaturesToPublish)
        {
            v.clear();
            b->getLimits(v);
            _bestFeatures.push_back(std::make_pair(v, s->getWeight()));
        }
    }


    int max = 0;
    _worst = 0;


    std::shared_ptr<adaboost::WeakClassifier> wc;

    for(size_t i = 0; i < _weakClassifiers.size(); i++)
    {
        wc = _weakClassifiers[i];
        {
            if(wc->getMarks() > max && !wc->isSelected())
            {
                _worst = i;
                max = wc->getMarks();
            }
        }
        wc->reset();
    }
}

void StrongClassifier::update(const cv::Rect &target, int response, const std::pair<cv::Mat, cv::Mat> &integral)
{

    for(size_t i = 0; i < _weakClassifiers.size(); i++)
    {
        std::shared_ptr<adaboost::WeakClassifier> wc = _weakClassifiers[i];
        wc->train(target, response, integral);
    }

    float lambda = 1.0f;

    for(size_t j = 0; j < _selectors.size(); j++)
    {
        std::shared_ptr<Selector> s = _selectors[j];
        s->selectBest(_weakClassifiers, lambda);
    }

    int max = 0;
    _worst = 0;

    for(size_t i = 0; i < _weakClassifiers.size(); i++)
    {
        std::shared_ptr<adaboost::WeakClassifier> wc = _weakClassifiers[i];
        if(wc->getMarks() > max && !wc->isSelected())
        {
            _worst = i;
            max = wc->getMarks();
        }
        wc->reset();
    }
}


void StrongClassifier::updateTarget(const cv::Size& size)
{
    for(size_t i = 0; i < _weakClassifiers.size(); i++)
	{
        std::shared_ptr<adaboost::WeakClassifier> wc = _weakClassifiers[i];
		wc->adjustSize(size);
	}
}

std::vector<std::pair<std::vector<std::pair<int, int> >, float> > StrongClassifier::getBestFeatures()
{
	return _bestFeatures;
}

float StrongClassifier::predict(const cv::Rect& target, const std::pair<cv::Mat, cv::Mat>& integral)
{
	float sum = 0.0f;
    std::shared_ptr<Selector> s;
	for(size_t i = 0; i < _selectors.size(); i++)
	{
        s = _selectors[i];
        sum += s->predict(target, integral);
	}
	return sum/_selectors.size();
}

float StrongClassifier::predict(const cv::Point& location, const std::pair<cv::Mat, cv::Mat>& integral)
{
	float sum = 0.0f;
    std::shared_ptr<Selector> s;
	for(size_t i = 0; i < _selectors.size(); i++)
	{
        s = _selectors[i];
		sum += s->predict(location, integral);
	}
	return sum/_selectors.size();
}

} /* namespace adaboost */
