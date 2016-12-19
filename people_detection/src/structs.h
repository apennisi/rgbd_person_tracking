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



#ifndef STRUCTS_H
#define STRUCTS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


namespace fpdw
{
namespace structs
{

enum Type{FPDW_MAX, FPDW_MAXG, FPDW_MS, FPDW_COVER};
enum OvrDnm{FPDW_UNION, FPDW_MIN};
enum ColorSpace{FPDW_HSV, FPDW_LUV, FPDW_RGB, FPDW_ORIG, FPDW_GRAY};
enum PadWith{FPDW_NONE, FPDW_REPLICATE};

struct Nms
{
public:
    void print()
    {
        std::cout << "\tNms" << std::endl;
        std::cout << "\t\ttype: " << type << std::endl;
        std::cout << "\t\tovrDnm: " << ovrDnm << std::endl;
        std::cout << "\t\toverlap: " << overlap << std::endl;
    }
    Nms &operator=(const Nms& other)
    {
        if(this==&other)
        {return *this;}

        type = other.type;
        ovrDnm = other.ovrDnm;
        overlap = other.overlap;

        return *this;
    }

public:
    Type type;
    OvrDnm ovrDnm;
    float overlap;
};

struct Color
{
public:
    Color &operator=(const Color& other)
    {
        if(this==&other)
        {return *this;}

        enable = other.enable;
        smooth = other.smooth;
        colorSpace = other.colorSpace;

        return *this;
    }

public:
    bool enable;
    bool smooth;
    ColorSpace colorSpace;
};

struct GradMag
{
public:
    GradMag &operator=(const GradMag& other)
    {
        if(this==&other)
        {return *this;}

        enable = other.enable;
        colorChn = other.colorChn;
        normRad = other.normRad;
        normConst = other.normConst;
        full = other.full;

        return *this;
    }
public:
    bool enable;
    int colorChn;
    int normRad;
    float normConst;
    bool full;
};

struct GradHist
{
public:
    GradHist &operator=(const GradHist& other)
    {
        if(this==&other)
        {return *this;}

        enable = other.enable;
        binSize = other.binSize;
        nOrients = other.nOrients;
        softBin = other.softBin;
        useHog = other.useHog;
        clipHog = other.clipHog;

        return *this;
    }
public:
    bool enable;
    int binSize;
    int nOrients;
    int softBin;
    bool useHog;
    float clipHog;

};

struct Chns
{
public:
    Chns &operator=(const Chns &other)
    {
        if(this==&other)
        {return *this;}

        shrink = other.shrink;
        complete = other.complete;
        pColor = other.pColor;
        pGradMag = other.pGradMag;
        pGradHist = other.pGradHist;

        return *this;
    }

public:
    int shrink;
    int complete;
    Color pColor;
    GradMag pGradMag;
    GradHist pGradHist;
};

struct Tree
{
public:
    Tree &operator=(const Tree &other)
    {
        if(this==&other)
        {return *this;}

        nBins = other.nBins;
        maxDepth = other.maxDepth;
        minWeight = other.minWeight;
        fracFtrs = other.fracFtrs;
        nThreads = other.nThreads;

        return *this;
    }

    void print()
    {
        std::cout << "\t\tTree" << std::endl;
        std::cout << "\t\t\tnBins: " << nBins << std::endl;
        std::cout << "\t\t\tmaxDepth: " << maxDepth << std::endl;
        std::cout << "\t\t\tminWeight: " << minWeight << std::endl;
        std::cout << "\t\t\tfracFtrs: " << fracFtrs << std::endl;
        std::cout << "\t\t\tnThreads: " << nThreads << std::endl;
    }

public:
    int nBins;
    int maxDepth;
    float minWeight;
    float fracFtrs;
    int nThreads;
};

struct Boost
{
public:
    Boost &operator=(const Boost &other)
    {
        if(this==&other)
        {return *this;}

        pTree = other.pTree;
        nWeak = other.nWeak;
        discrete = other.discrete;
        verbose = other.verbose;

        return *this;
    }

    void print()
    {
        pTree.print();
        std::cout << "\t\tnWeek: " << nWeak << std::endl;
        std::cout << "\t\tdiscrete: " << discrete << std::endl;
        std::cout << "\t\tverbose: " << verbose << std::endl;
    }

public:
    Tree pTree;
    int nWeak;
    int discrete;
    int verbose;
};

struct Cls
{
public:
    Cls &operator=(const Cls &other)
    {
        if(this==&other)
        {return *this;}

        fids = other.fids;
        thrs = other.thrs;
        child = other.child;
        hs = other.hs;
        weights = other.weights;
        depth = other.depth;
        errs = other.errs;
        losses = other.losses;
        treeDepth = other.treeDepth;
        fidsSize = other.fidsSize;

        return *this;
    }

    void print()
    {
        std::cout << "****CLS****" << std::endl;
        std::cout << "fids" << std::endl;

        for(const auto &it : fids)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }


        std::cout << "thrs" << std::endl;
        for(const auto &it : thrs)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }


        std::cout << "child" << std::endl;
        for(const auto &it : child)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }

        std::cout << "hs" << std::endl;
        for(const auto &it : hs)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }

        std::cout << "weights" << std::endl;
        for(const auto &it : weights)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }

        std::cout << "depth" << std::endl;
        for(const auto &it : depth)
        {
            std::cout <<  it << " ";
            std::cout << std::endl;
        }

        std::cout << "errs" << std::endl;
        for(const auto &it : errs)
        {
            std::cout << it << " ";
        }
        std::cout << std::endl;

        std::cout << "losses" << std::endl;
        for(const auto &it : losses)
        {
            std::cout << it << " ";
        }
        std::cout << std::endl;

        std::cout << "treeDepth: " << treeDepth << std::endl;
    }

public:
    std::vector<int> fids;
    std::vector<float> thrs;
    std::vector<int> child;
    std::vector<float> hs;
    std::vector<float> weights;
    std::vector<int> depth;
    std::vector<float> errs;
    std::vector<float> losses;
    cv::Size fidsSize;
    int treeDepth;
};

struct Pyramid
{
public:
    Pyramid &operator=(const Pyramid &other)
    {
        if(this==&other)
        {return *this;}

        pChns = other.pChns;
        nPerOct = other.nPerOct;
        nOctup = other.nOctup;
        nApprox = other.nApprox;
        lambdas = other.lambdas;
        pad = other.pad;
        minDs = other.minDs;
        smooth = other.smooth;
        concat = other.concat;
        complete = other.complete;

        return *this;
    }

    void print()
    {
        std::cout << "\tPYRAMID:" << std::endl;
        std::cout << "\tchns" << std::endl;
        std::cout << "\t\tshrink: " << pChns.shrink << std::endl;
        std::cout << "\t\tcolor: " << pChns.pColor.enable << " "
                  <<  pChns.pColor.smooth << " " <<
                      pChns.pColor.colorSpace << std::endl;
        std::cout << "\t\tgradmag: " << pChns.pGradMag.enable << " " <<
                     pChns.pGradMag.colorChn << " " << pChns.pGradMag.normRad <<
                     " " << pChns.pGradMag.normConst << " " << pChns.pGradMag.full << std::endl;

        std::cout << "\t\tgradhist: " << pChns.pGradHist.enable << " "
                  << pChns.pGradHist.nOrients << " " <<
                     " "  << pChns.pGradHist.softBin << " " << pChns.pGradHist.useHog << " "
                  << pChns.pGradHist.clipHog << std::endl;

        std::cout << "\t\tcomplete: " << pChns.complete << std::endl;

        std::cout << "\tperoct: " << nPerOct << std::endl;
        std::cout << "\toctup: " << nOctup << std::endl;
        std::cout << "\tapprox: " << nApprox << std::endl;

        std::cout << "\tlambdas: " << std::endl;
        for(const auto &it : lambdas)
        {
            std::cout << it << std::endl;
        }

        std::cout << "\tpad: " << std::endl;
        for(const auto &it : pad)
        {
            std::cout << it << std::endl;
        }
        std::cout << "\tminds: " << minDs << std::endl;

        std::cout << "\tsmooth: " << smooth << std::endl;
        std::cout << "\tconcat: " << concat << std::endl;
        std::cout << "\tcomplete: " << complete << std::endl;
        std::cout << "\n\n";
    }

public:
    Chns pChns;
    int nPerOct;
    int nOctup;
    int nApprox;
    std::vector<float> lambdas;
    std::vector<int> pad;
    cv::Size minDs;
    bool smooth;
    bool concat;
    bool complete;
};

struct Options
{
public:
    Options &operator=(const Options &other)
    {
        if(this==&other)
        {return *this;}

        pPyramid = other.pPyramid;
        pBoost = other.pBoost;
        filters = other.filters;
        pNms = other.pNms;
        modelDs = other.modelDs;
        modelDsPad = other.modelDsPad;
        stride = other.stride;
        cascThr = other.cascThr;
        cascCal = other.cascCal;
        nWeak = other.nWeak;
        seed = other.seed;
        nPos = other.nPos;
        nNeg = other.nNeg;
        nPerNeg = other.nPerNeg;
        nAccNeg = other.nAccNeg;
        winSave = other.winSave;
        pLoad = other.pLoad;
        flip = other.flip;

        return *this;
    }

    void print()
    {
        std::cout << "****OPTIONS****" << std::endl;
        pPyramid.print();
        std::cout << "\tmodelDs: " << std::endl;
        for(const auto &it : modelDs)
        {
            std::cout << it << std::endl;
        }
        std::cout << "\tmodelDsPad: " << std::endl;
        for(const auto &it : modelDsPad)
        {
            std::cout << it << std::endl;
        }
        pNms.print();
        std::cout << "\tstride: " << stride << std::endl;
        std::cout << "\tcascthr: " << cascThr << std::endl;
        std::cout << "\tcasccal: " << cascCal << std::endl;
        std::cout << "\tnweak: " << nWeak[0] << " " << nWeak[1] <<
                     " " << nWeak[2] << " " << nWeak[3] << std::endl;
        pBoost.print();
        std::cout << "\tseed: " << seed << std::endl;
        std::cout << "\tsqyarify: " << pLoad << std::endl;
        std::cout << "\tpos: " << nPos << std::endl;
        std::cout << "\tneg: " << nNeg << std::endl;
        std::cout << "\tperNeg: " << nPerNeg << std::endl;
        std::cout << "\taccneg: " << nAccNeg << std::endl;
        std::cout << "\tflip: " << flip << std::endl;
        std::cout << "\twinsave: " << winSave << std::endl;
    }

public:
    Pyramid pPyramid;
    Boost pBoost;
    std::vector<float> filters;
    Nms pNms;
    std::vector<int> modelDs;
    std::vector<int> modelDsPad;
    uint stride;
    int cascThr;
    float cascCal;
    std::vector<int> nWeak;
    int seed;
    int nPos;
    int nNeg;
    int nPerNeg;
    int nAccNeg;
    int winSave;
    std::string posGtDir;
    std::string posImgDir;
    std::string negImgDir;
    std::string posWinDir;
    std::string negWinDir;
    cv::Size2f pLoad;
    int flip;
};

struct BBox
{
public:
    BBox &operator=(const BBox &_other)
    {
        if(this==&_other)
        {return *this;}

        bbox = _other.bbox;
        conf = _other.conf;

        return *this;
    }

public:
    cv::Rect bbox;
    float conf;
};

}
}

#endif
