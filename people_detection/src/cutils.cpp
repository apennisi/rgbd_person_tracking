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


#include "cutils.h"

void *alMalloc(size_t size, int alignment)
{
    const size_t pSize = sizeof(void*), a = alignment-1;
    void *raw = (void*) wrMalloc(size + a + pSize);
    void *aligned = (void*) (((size_t) raw + pSize + a) & ~a);
    *(void**) ((size_t) aligned-pSize) = raw;
    return aligned;
}


void alFree(void *aligned)
{
    void* raw = *(void**)((char*)aligned-sizeof(void*));
    wrFree(raw);
}


cv::Mat cvMatToMat(const cv::Mat &_img)
{
    cv::Mat image;
    const uint &channels = _img.channels();
    std::vector<cv::Mat> imgChannels;
    cv::split(_img, imgChannels);

    for(uint i = 0; i < channels; ++i)
    {
        imgChannels[i] = imgChannels[i].t();
    }

    image = imgChannels[0];
    for(uint i = 1; i < channels; ++i)
    {
        cv::vconcat(image, imgChannels[i], image);
    }
    return image;
}

///TODO FARE UNICA FUNZIONE MATNN
cv::Mat matToCvMat6x(float *O, const cv::Size &_sz)
{
    cv::Mat_<cv::Vec6f> out(_sz);

    int double_area = 2*_sz.area();

    cv::Vec6f vec;
    for(uint j = 0; j < _sz.width; ++j)
    {
        for(uint i = 0; i < _sz.height; ++i)
        {
            vec[0] = O[(j*_sz.height + i)];
            vec[1] = O[(j*_sz.height + i) + _sz.area()];
            vec[2] = O[(j*_sz.height + i) + double_area];
            vec[3] = O[(j*_sz.height + i) + double_area + _sz.area()];
            vec[4] = O[(j*_sz.height + i) + double_area + double_area];
            vec[5] = O[(j*_sz.height + i) + double_area + double_area + _sz.area()];
            out.at<cv::Vec6f>(i, j) = vec;
        }

    }

    return out;

}

cv::Mat matToCvMat3x(float *O, const cv::Size &_sz)
{
    cv::Mat out(_sz, CV_32FC3);
    int double_area = 2*_sz.area();
    cv::Vec3f vec;

    for(uint j = 0; j < _sz.width && O; ++j)
    {
        for(uint i = 0; i < _sz.height; ++i)
        {
            vec[0] = O[(j*_sz.height + i)];
            vec[1] = O[(j*_sz.height + i) + _sz.area()];
            vec[2] = O[(j*_sz.height + i) + double_area];
            out.at<cv::Vec3f>(i, j) = vec;
        }

    }

    return out;

}

cv::Mat matToCvMat1x(float *O, const cv::Size &_sz)
{
    cv::Mat out(_sz, CV_32FC1);

    for(uint j = 0; j < _sz.width; ++j)
    {
        for(uint i = 0; i < _sz.height; ++i)
        {
            out.at<float>(i, j) = O[(j*_sz.height + i)];
        }

    }

    return out;

}

int channels(int type)
{
    int chn;
    uchar depth = type & CV_MAT_DEPTH_MASK;

    switch ( type )
    {
        case CV_8UC1:
        case CV_8SC1:
        case CV_16UC1:
        case CV_16SC1:
        case CV_32SC1:
        case CV_32FC1:
        case CV_64FC1:
            chn = 1;
            break;
        case CV_8UC3:
        case CV_8SC3:
        case CV_16UC3:
        case CV_16SC3:
        case CV_32SC3:
        case CV_32FC3:
        case CV_64FC3:
            chn = 3;
            break;
        default:
            chn = 6;
    }
    return chn;
}
