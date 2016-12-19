/*
 *  The FASTEST PEDESTRIAN DETECTOR IN THE WEST (FPDW)
 *  Copyright 2015 Andrea Pennisi
 *
 *  This file is part of AT and it is distributed under the terms of the
 *  GNU Lesser General Public License (Lesser GPL)
 *
 *
 *
 *  FPDW is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  FPDW is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with FPDW.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  FPDW has been written by Andrea Pennisi
 *
 *  Please, report suggestions/comments/bugs to
 *  andrea.pennisi@gmail.com
 *
 */


#ifndef CUTILS_H
#define CUTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>

inline void wrError(const char *errormsg)
{
    throw errormsg;
}

inline void* wrCalloc( size_t num, size_t size )
{
    return calloc(num,size);
}
inline void* wrMalloc( size_t size )
{
    return malloc(size);
}
inline void wrFree( void * ptr )
{
    free(ptr);
}

void alFree(void *aligned);
void *alMalloc(size_t size, int alignment);
cv::Mat cvMatToMat(const cv::Mat &_img);
cv::Mat matToCvMat6x(float *O, const cv::Size &_sz);
cv::Mat matToCvMat3x(float *O, const cv::Size &_sz);
cv::Mat matToCvMat1x(float *O, const cv::Size &_sz);
int channels(int type);

#endif
