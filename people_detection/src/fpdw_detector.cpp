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


#include "fpdw_detector.h"

using namespace fpdw;
using namespace fpdw::detector;

FPDWDetector::FPDWDetector(const std::string &_xml_file, const float &_conf)
    : m_confidence(_conf)
{
    m_cls.load(_xml_file);
    m_clf = m_cls.cls();
    m_modelDsPad = m_cls.opts().modelDsPad;
    m_modelDs = m_cls.opts().modelDs;
    m_stride = m_cls.opts().stride;
    m_cascThr = m_cls.opts().cascThr;
    m_nPerOct = m_cls.opts().pPyramid.nPerOct;
    m_nOctUp = m_cls.opts().pPyramid.nOctup;
    m_minDs = m_cls.opts().pPyramid.minDs;
    m_shrink = m_cls.opts().pPyramid.pChns.shrink;
    m_shrink_inv = 1 / float(m_shrink);
    m_shift.resize(m_modelDsPad.size());
    init_detector = true;
    for(uint i = 0; i < m_modelDsPad.size(); ++i)
    {
        m_shift[i] = float(m_modelDsPad[i] - m_cls.opts().modelDs[i]) / 2. - m_cls.opts().pPyramid.pad[i];
    }

    //CLASSIFIER PARAMETERS
    modelHt = (int) m_modelDsPad[0];
    modelWd = (int) m_modelDsPad[1];
    stride = (int) m_stride;
    cascThr = (float) m_cascThr;

    thrs = (float*) m_clf.thrs.data();
    hs = (float*) m_clf.hs.data();
    fids = (int*) m_clf.fids.data();
    child = (int*) m_clf.child.data();
    treeDepth = m_clf.treeDepth;

    nTreeNodes =  m_clf.fidsSize.height;
    nTrees = m_clf.fidsSize.width;

    stride_shrink = stride/m_shrink;

    endWd = modelWd*m_shrink_inv;
    endHt = modelHt*m_shrink_inv;

    //NMS
    nms = Nms(m_cls.opts().pNms);
}

FPDWDetector::FPDWDetector(const Classifier &_cls)
{
    m_cls = _cls;
    m_clf = _cls.cls();
    m_modelDsPad = _cls.opts().modelDsPad;
    m_modelDs = _cls.opts().modelDs;
    m_stride = _cls.opts().stride;
    m_cascThr = _cls.opts().cascThr;
    m_nPerOct = _cls.opts().pPyramid.nPerOct;
    m_nOctUp = _cls.opts().pPyramid.nOctup;
    m_minDs = _cls.opts().pPyramid.minDs;
    m_shrink = _cls.opts().pPyramid.pChns.shrink;
    m_shrink_inv = 1 / float(m_shrink);
    m_shift.resize(m_modelDsPad.size());
    init_detector = true;
    for(uint i = 0; i < m_modelDsPad.size(); ++i)
    {
        m_shift[i] = float(m_modelDsPad[i] - _cls.opts().modelDs[i]) / 2. -_cls.opts().pPyramid.pad[i];
    }

    //CLASSIFIER PARAMETERS
    modelHt = (int) m_modelDsPad[0];
    modelWd = (int) m_modelDsPad[1];
    stride = (int) m_stride;
    cascThr = (float) m_cascThr;

    thrs = (float*) m_clf.thrs.data();
    hs = (float*) m_clf.hs.data();
    fids = (int*) m_clf.fids.data();
    child = (int*) m_clf.child.data();
    treeDepth = m_clf.treeDepth;

    nTreeNodes =  m_clf.fidsSize.height;
    nTrees = m_clf.fidsSize.width;

    stride_shrink = stride/m_shrink;

    endWd = modelWd*m_shrink_inv;
    endHt = modelHt*m_shrink_inv;

    //NMS
    nms = Nms(m_cls.opts().pNms);
}

void FPDWDetector::process(const cv::Mat &_img)
{
    _img.copyTo(m_img);

    if(init_detector)
    {
        m_img_size = m_img.size();
        init();
    }
    m_data.clear();
    m_bboxes.clear();
    m_data.resize(nScales);

    //BUILD PYRAMID
    chnsPyramid(m_cls.opts().pPyramid);

    cv::Mat bbs, image;
    for(uint i = 0; i < nScales; ++i)
    {
        image = detect(i);
        if(image.rows == 0) continue;
        if(!bbs.empty())
        {
            cv::vconcat(bbs, image, bbs);
        }
        else
        {
            bbs = image.clone();
        }
    }

    std::vector<cv::Rect> bbox;
    std::vector<float> conf;
    nms.process(bbox, conf, bbs);

    for(int i = 0; i < bbox.size(); ++i)
    {
        if(conf[i] > m_confidence)
        {
            m_bboxes.push_back(bbox[i]);
        }
    }
}

void FPDWDetector::chnsPyramid(const fpdw::structs::Pyramid &_pyr)
{
    cv::Mat luv;
    float s;
    cv::Size sz;
    cv::Mat scaledImg;
    float iR;
    int iA;
    float scale_iA_iR;
    float ratio;
    std::vector<cv::Mat> internal_data;
    cv::Mat image_data;
    utils::RgbConvertion::process(m_img, _pyr.pChns.pColor.colorSpace, luv);
    
    //Compute image pyramid
    int isr;
    for(const auto &i : isR)
    {
        isr = i - 1;
        s = m_scales.at(isr);
        sz.width = cvRound(m_img_size.width * s * m_shrink_inv) * m_shrink;
        sz.height = cvRound(m_img_size.height * s * m_shrink_inv) * m_shrink;
        if(sz.width == m_img_size.width && sz.height == m_img_size.height)
        {
            scaledImg = luv.clone();
        }
        else
        {
            utils::ImResample<float>::resample(luv, sz, 1, luv.channels(), scaledImg);
        }

        if(s == .5 && (m_cls.opts().pPyramid.nApprox >0 || m_nPerOct == 1))
        {
            luv = scaledImg.clone();
        }

        chnsCompute(scaledImg, _pyr.pChns, isr);
    }

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) private(internal_data, iA, iR, sz, scale_iA_iR, ratio, image_data)
    for(uint i = 0; i < isA.size(); ++i)
    {
        iA = isA.at(i) - 1;
        iR = isN.at(iA) - 1;
        sz = cv::Size(cvRound(m_img_size.width*m_scales.at(iA)*m_shrink_inv),
                      cvRound(m_img_size.height*m_scales.at(iA)*m_shrink_inv));
        scale_iA_iR = m_scales.at(iA) / float(m_scales.at(iR));
        internal_data.clear();
        for(uint j = 0; j < nTypes; ++j)
        {
            ratio = std::pow(scale_iA_iR, -_pyr.lambdas.at(j));
            utils::ImResample<float>::resample(m_data.at(iR).at(j), sz, ratio, m_data.at(iR).at(j).channels(), image_data);
            internal_data.push_back(image_data.clone());
        }
        #pragma omp critical
        {
            m_data.at(iA) = internal_data;
        }
    }

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
    for(uint i = 0; i < nScales; ++i)
    {
        for(uint j = 0; j < nTypes; ++j)
        {
            utils::Convolution::convTri1(m_data.at(i).at(j), _pyr.smooth, m_data.at(i).at(j));
        }
    }

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
    for(uint i = 0; i < nScales; ++i)
    {
        for(uint j = 0; j < nTypes; ++j)
        {
            utils::ImPad<float>::impad(m_data.at(i).at(j), _pyr.pad, m_padWith.at(j), m_data.at(i).at(j), m_shrink_inv);
        }
    }

}

void FPDWDetector::chnsCompute(const cv::Mat &_img, const fpdw::structs::Chns &_chns,
                               const int &_isr)
{
    nTypes = 0;
    cv::Size img_size = _img.size();
    cv::Mat img = _img.clone();
    std::vector<cv::Mat> temp_data;
    bool smooth = _chns.pColor.smooth;

    cv::Size cr(img_size.width % m_shrink, img_size.height % m_shrink);
    if(cr.width != 0 || cr.height != 0)
    {
        img_size.height = img_size.height - cr.height;
        img_size.width = img_size.width - cr.width;
    }

    img_size.height *= m_shrink_inv;
    img_size.width *= m_shrink_inv;

    cv::Mat cImg;
    utils::Convolution::convTri1(img, smooth, cImg);

    cv::Mat data;
    if(img_size.width != cImg.size().width && img_size.height != cImg.size().height)
    {
        utils::ImResample<float>::resample(cImg, img_size, 1, img.channels(), data);
    }
    else
    {
        data = cImg.clone();
    }

    nTypes++;
    temp_data.push_back(data.clone());
    m_padWith.push_back(structs::PadWith::FPDW_REPLICATE);

    fpdw::structs::GradMag p = _chns.pGradMag;

    cv::Mat O;
    if(_chns.pGradHist.enable)
    {
        gradient.gradMag(cImg, p, 2);
        O = gradient.O().clone();
    }
    else
    {
        gradient.gradMag(cImg, p, 1);
    }
    cv::Mat M = gradient.M().clone();
    cv::Mat S;
    utils::Convolution::convTri(M, 5, S);

    M = gradient.gradMagNorm(M, S, p);

    if(p.enable)
    {
        if(img_size.width != M.size().width && img_size.height != M.size().height)
        {
            utils::ImResample<float>::resample(M, img_size, 1, M.channels(), data);
        }
        else
        {
            data = M.clone();
        }

        nTypes++;
        temp_data.push_back(data.clone());
        m_padWith.push_back(structs::PadWith::FPDW_NONE);
    }


    if(p.enable)
    {
        int binSize = m_shrink;
        cv::Mat H = gradient.gradientHist(M, O, binSize, _chns.pGradHist, p.full);
        if(img_size.width != H.size().width && img_size.height != H.size().height)
        {
            utils::ImResample<float>::resample(H, img_size, 1, H.channels(), data);
        }
        else
        {
            data = H.clone();
        }

        nTypes++;
        temp_data.push_back(data.clone());
        m_padWith.push_back(structs::PadWith::FPDW_NONE);
    }

    m_data.at(_isr) = temp_data;
}

void FPDWDetector::getChild(float *chns1, uint *cids, int *fids, float *thrs, int offset, int &k0, int &k)
{
    float ftr = chns1[cids[fids[k]]];
    k = (ftr<thrs[k]) ? 1 : 2;
    k0=k+=k0*2; k+=offset;
}

void FPDWDetector::init()
{
    getScale();
    isA.resize(nScales);
    isN.resize(nScales);

    for(uint i = 1; i < nScales; i += m_cls.opts().pPyramid.nApprox + 1)
    {
        isR.push_back(i);
    }

    std::iota(isA.begin(), isA.end(), 1);
    std::iota(isN.begin(), isN.end(), 1);

    int i = 0;
    for(const auto isr : isR)
    {
        isA.erase(isA.begin() + ((isr - 1) - i++));
    }

    std::vector<int> v_j;
    v_j.push_back(0);
    for(uint i = 0; i < isR.size() - 1; ++i)
    {
        v_j.push_back(std::floor((isR.at(i) + isR.at(i+1)) * .5));
    }
    v_j.push_back(nScales);

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
    for(uint i = 0; i < isR.size(); ++i)
    {
        for(uint j = v_j.at(i); j < v_j.at(i + 1); ++j)
        {
            isN.at(j) = isR.at(i);
        }
    }

    init_detector = false;
}

cv::Mat FPDWDetector::detect(const int &_i)
{
    // get inputs
    int nChns;
    cv::Mat image = vectorToCvMat(m_data.at(_i), nChns);
    float *chns = image.ptr<float>(0);

    // get dimensions and constants
    const int &height = m_data.at(_i).at(0).size().height;
    const int &width = m_data.at(_i).at(0).size().width;
    const int &area = height * width;
    const int &height1 = (int) ceil((float)(height*m_shrink-modelHt+1)/stride);
    const int &width1 = (int) ceil((float)(width*m_shrink-modelWd+1)/stride);


    // construct cids array
    int nFtrs = modelHt*m_shrink_inv*modelWd*m_shrink_inv*nChns;

    uint *cids = new uint[nFtrs];
    int m = 0;

    for( int z = 0; z < nChns; ++z )
    {
        for( int c = 0; c < endWd; ++c )
        {
            for( int r = 0; r < endHt; ++r)
            {
                cids[m++] = z*area + c*height + r;
            }
        }
    }

    // apply classifier to each patch
    std::vector<int> rs, cs;
    std::vector<float> hs1;
    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) shared(rs, cs, hs1)
    for( int c=0; c<width1; c++ )
    {
        for( int r=0; r<height1; r++ )
        {
            float h=0, *chns1=chns+(r*stride_shrink) + (c*stride_shrink)*height;
            if( treeDepth==1 )
            {
                // specialized case for treeDepth==1
                for( int t = 0; t < nTrees; t++ )
                {
                    int offset=t*nTreeNodes, k=offset, k0=0;
                    getChild(chns1,cids,fids,thrs,offset,k0,k);
                    h += hs[k];
                    if( h<=cascThr ) break;
                }
            } else if( treeDepth==2 )
            {
                // specialized case for treeDepth==2
                for( int t = 0; t < nTrees; t++ )
                {
                    int offset=t*nTreeNodes, k=offset, k0=0;
                    getChild(chns1,cids,fids,thrs,offset,k0,k);
                    getChild(chns1,cids,fids,thrs,offset,k0,k);
                    h += hs[k];
                    if( h<=cascThr ) break;
                }
            } else if( treeDepth>2)
            {
                // specialized case for treeDepth>2
                for( int t = 0; t < nTrees; t++ )
                {
                    int offset=t*nTreeNodes, k=offset, k0=0;
                    for( int i=0; i<treeDepth; i++ )
                        getChild(chns1,cids,fids,thrs,offset,k0,k);
                    h += hs[k];
                    if( h<=cascThr ) break;
                }
            } else {
                // general case (variable tree depth)
                for( int t = 0; t < nTrees; t++ )
                {
                    int offset=t*nTreeNodes, k=offset, k0=k;
                    while( child[k] )
                    {
                        float ftr = chns1[cids[fids[k]]];
                        k = (ftr<thrs[k]) ? 1 : 0;
                        k0 = k = child[k0]-k+offset;
                    }
                    h += hs[k];
                    if( h<=cascThr ) break;
                }
            }
            if(h>cascThr)
            {
                #pragma omp critical
                {
                    cs.push_back(c);
                    rs.push_back(r);
                    hs1.push_back(h);
                }
            }
        }
    }
    delete [] cids; m=cs.size();

    cv::Mat_<float> output(cv::Size(5, m));

    #pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() )
    for(uint i = 0; i < m; ++i)
    {
        output.at<float>(i, 0) = ((cs[i]*stride) + float(m_shift[1])) / float(m_scaleHw.at(_i).y);
        output.at<float>(i, 1) = ((rs[i]*stride) + float(m_shift[0])) / float(m_scaleHw.at(_i).x);
        output.at<float>(i, 2) = m_modelDs[1] / float(m_scales.at(_i));
        output.at<float>(i, 3) = m_modelDs[0] / float(m_scales.at(_i));
        output.at<float>(i, 4) = hs1[i];
    }

    return output;
}

cv::Mat FPDWDetector::vectorToCvMat(const std::vector<cv::Mat> &_v, int &nChannels)
{
    cv::Mat output;
    output = cvMatToMat(_v[0]);
    nChannels = _v.at(0).channels();
    for(uint i = 1; i < _v.size(); ++i)
    {
        nChannels += _v.at(i).channels();
        cv::vconcat(output, cvMatToMat(_v.at(i)), output);
    }
    return output;
}

void FPDWDetector::getScale()
{
    nScales = std::floor(m_nPerOct *
                             (m_nOctUp + std::log2( std::min(m_img_size.width / float(m_minDs.width),
                                                             m_img_size.height / float(m_minDs.height)))) + 1 );

    std::vector<float> scales;

    for(uint i = 0; i  < nScales; ++i)
    {
        scales.push_back(std::pow(2, -float(i)/float(m_nPerOct) + m_nOctUp));
    }


    int d_0, d_1;
    if(m_img_size.height < m_img_size.width)
    {
        d_0 = m_img_size.height;
        d_1 = m_img_size.width;
    }
    else
    {
        d_0 = m_img_size.width;
        d_1 = m_img_size.height;
    }

    float s, minElem;
    float s_0, s_1, val_0, val_1;
    float elem;
    std::vector<float> ss;
    int indexMin;


    for(uint i = 0; i < nScales; ++i)
    {
        s = scales.at(i);
        ss.clear();
        s_0 = (cvRound(d_0 * s * m_shrink_inv) * m_shrink - .25 * m_shrink)/float(d_0);
        s_1 = (cvRound(d_0 * s * m_shrink_inv) * m_shrink + .25 * m_shrink)/float(d_0);
        for(float j = 0.; j <= 1.0; j+=0.01)
        {
            ss.push_back(j*(s_1-s_0) + s_0);
        }

        val_0 = d_0*ss.at(0);
        val_0 = std::abs(val_0 - cvRound(float(val_0) * m_shrink_inv)*m_shrink);
        val_1 = d_1*ss.at(0);
        val_1 = std::abs(val_1 - cvRound(float(val_1) * m_shrink_inv)*m_shrink);

        minElem = std::max(val_0, val_1);
        indexMin = 0;

        for(int j = 1; j < ss.size(); ++j)
        {
            val_0 = d_0*ss.at(j);
            val_0 = std::abs(val_0 - cvRound(float(val_0) * m_shrink_inv)*m_shrink);
            val_1 = d_1*ss.at(j);
            val_1 = std::abs(val_1 - cvRound(float(val_1) * m_shrink_inv)*m_shrink);
            elem = std::max(val_0, val_1);
            if(elem < minElem)
            {
                minElem = elem;
                indexMin = j;
            }
        }

        scales.at(i) = ss.at(indexMin);
    }

    bool check = true;
    std::vector<bool> checks(scales.size(), true);
    for(uint i = 0; i < (scales.size() - 1); ++i)
    {
        if(scales.at(i) == scales.at(i + 1))
        {
            check = false;
            checks.at(i) = false;
        }
    }

    if(check)
    {
        m_scales = scales;
    }
    else
    {
        for(uint i = 0; i < checks.size(); ++i)
        {
            if(checks.at(i))
            {
                m_scales.push_back(scales.at(i));
            }
        }
    }

    cv::Point2f p;

    for(uint i = 0; i < m_scales.size(); ++i)
    {
        p.x = cvRound(m_img_size.height * m_scales.at(i) * m_shrink_inv) * m_shrink/float(m_img_size.height);
        p.y = cvRound(m_img_size.width * m_scales.at(i) * m_shrink_inv) * m_shrink/float(m_img_size.width);
        m_scaleHw.push_back(p);
    }
}
