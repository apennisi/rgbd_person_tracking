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


#include "classifier.h"

using namespace fpdw;


Classifier &Classifier::operator=(const Classifier &obj)
{
    if(this==&obj)
    {return *this;}

    m_opts = obj.opts();
    m_cls = obj.cls();

    return *this;
}


void Classifier::load(const std::string &_filename)
{
    try
    {
        file.open(_filename);
    }
    catch(...)
    {
        std::cerr << "Error: Impossible to open: " << _filename << std::endl;
        exit(-1);
    }

    file.close();

    using boost::property_tree::ptree;
    ptree pt;

    try
    {
        read_xml(_filename, pt);
    }
    catch(...)
    {
        std::cerr << "Error: Impossible to read the file: " << _filename << std::endl;
        exit(-1);
    }

    //Color
    m_opts.pPyramid.pChns.shrink = pt.get<int>("classifier.opts.pyramid.chns.shrink");
    m_opts.pPyramid.pChns.pColor.enable = pt.get<bool>("classifier.opts.pyramid.chns.color.enabled");
    m_opts.pPyramid.pChns.pColor.smooth = pt.get<bool>("classifier.opts.pyramid.chns.color.smooth");
    const int &cl = pt.get<int>("classifier.opts.pyramid.chns.color.colorspace");
    m_opts.pPyramid.pChns.pColor.colorSpace = static_cast<structs::ColorSpace>(cl);
    //GradMag
    m_opts.pPyramid.pChns.pGradMag.enable = pt.get<int>("classifier.opts.pyramid.chns.gradmag.enabled");
    m_opts.pPyramid.pChns.pGradMag.colorChn = pt.get<int>("classifier.opts.pyramid.chns.gradmag.colorchn");
    m_opts.pPyramid.pChns.pGradMag.normRad = pt.get<int>("classifier.opts.pyramid.chns.gradmag.normrad");
    m_opts.pPyramid.pChns.pGradMag.normConst = pt.get<float>("classifier.opts.pyramid.chns.gradmag.normconst");
    m_opts.pPyramid.pChns.pGradMag.full = pt.get<bool>("classifier.opts.pyramid.chns.gradmag.full");
    //GradHist
    m_opts.pPyramid.pChns.pGradHist.enable = pt.get<bool>("classifier.opts.pyramid.chns.gradhist.enabled");
    m_opts.pPyramid.pChns.pGradHist.nOrients = pt.get<int>("classifier.opts.pyramid.chns.gradhist.orients");
    m_opts.pPyramid.pChns.pGradHist.softBin = pt.get<int>("classifier.opts.pyramid.chns.gradhist.softbin");
    m_opts.pPyramid.pChns.pGradHist.useHog = pt.get<bool>("classifier.opts.pyramid.chns.gradhist.usehog");
    m_opts.pPyramid.pChns.pGradHist.clipHog = pt.get<float>("classifier.opts.pyramid.chns.gradhist.cliphog");

    m_opts.pPyramid.pChns.complete = pt.get<int>("classifier.opts.pyramid.chns.complete");
    m_opts.pPyramid.nPerOct = pt.get<int>("classifier.opts.pyramid.peroct");
    m_opts.pPyramid.nOctup = pt.get<int>("classifier.opts.pyramid.octup");
    m_opts.pPyramid.nApprox = pt.get<int>("classifier.opts.pyramid.approx");

    //LAMBDAS
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.opts.pyramid.lambdas"))
    {
        m_opts.pPyramid.lambdas.push_back(boost::lexical_cast<float>(v.second.data()));
    }

    //PAD
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.opts.pyramid.pad"))
    {
        m_opts.pPyramid.pad.push_back(boost::lexical_cast<int>(v.second.data()));
    }

    m_opts.pPyramid.minDs.height = pt.get<int>("classifier.opts.pyramid.minds.height");
    m_opts.pPyramid.minDs.width = pt.get<int>("classifier.opts.pyramid.minds.width");

    m_opts.pPyramid.smooth = pt.get<bool>("classifier.opts.pyramid.smooth");
    m_opts.pPyramid.concat = pt.get<bool>("classifier.opts.pyramid.concat");
    m_opts.pPyramid.complete = pt.get<bool>("classifier.opts.pyramid.complete");

    //MODELDS
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.opts.modelds"))
    {
        m_opts.modelDs.push_back(boost::lexical_cast<int>(v.second.data()));
    }

    //MODELDSPAD
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.opts.modeldspad"))
    {
        m_opts.modelDsPad.push_back(boost::lexical_cast<int>(v.second.data()));
    }

    const int &nms_type = pt.get<int>("classifier.opts.nms.type");
    m_opts.pNms.type = static_cast<structs::Type>(nms_type);

    m_opts.pNms.overlap = pt.get<float>("classifier.opts.nms.overlap");

    const int &nms_ovrdnm = pt.get<int>("classifier.opts.nms.ovrdnm");
    m_opts.pNms.ovrDnm = static_cast<structs::OvrDnm>(nms_ovrdnm);

    m_opts.stride = pt.get<uint>("classifier.opts.stride");
    m_opts.cascThr = pt.get<int>("classifier.opts.cascthr");
    m_opts.cascCal = pt.get<float>("classifier.opts.casccal");

    //MODELDSPAD
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.opts.weak"))
    {
        m_opts.nWeak.push_back(boost::lexical_cast<int>(v.second.data()));
    }

    //TREE
    m_opts.pBoost.pTree.nBins = pt.get<int>("classifier.opts.boost.tree.bins");
    m_opts.pBoost.pTree.maxDepth = pt.get<int>("classifier.opts.boost.tree.maxdepth");
    m_opts.pBoost.pTree.minWeight = pt.get<float>("classifier.opts.boost.tree.minweight");
    m_opts.pBoost.pTree.fracFtrs = pt.get<float>("classifier.opts.boost.tree.fracftrs");
    m_opts.pBoost.pTree.nThreads = pt.get<int>("classifier.opts.boost.tree.threads");

    m_opts.pBoost.nWeak = pt.get<int>("classifier.opts.boost.weak");
    m_opts.pBoost.discrete = pt.get<int>("classifier.opts.boost.discrete");
    m_opts.pBoost.verbose = pt.get<int>("classifier.opts.boost.verbose");

    m_opts.seed = pt.get<int>("classifier.opts.seed");

    m_opts.pLoad.height = pt.get<float>("classifier.opts.squarify.height");
    m_opts.pLoad.width = pt.get<float>("classifier.opts.squarify.width");

    try
    {
        const std::string &npos = pt.get<std::string>("classifier.opts.pos");
        if(npos.compare("Inf") == 0)
        {
            m_opts.nPos = std::numeric_limits<int>::infinity();
        }
    }
    catch(...)
    {
        m_opts.nPos = pt.get<int>("classifier.opts.pos");
    }

    m_opts.nNeg = pt.get<int>("classifier.opts.neg");
    m_opts.nPerNeg = pt.get<int>("classifier.opts.perneg");
    m_opts.nAccNeg = pt.get<int>("classifier.opts.accneg");
    m_opts.flip = pt.get<int>("classifier.opts.flip");
    m_opts.winSave = pt.get<int>("classifier.opts.winssave");

    std::string line;
    std::vector< std::vector<int> > fids;
    std::vector<int> line_vector;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.fids"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_vector.clear();
        for(auto i : token)
        {
            line_vector.push_back(boost::lexical_cast<int>(i));
        }
        fids.push_back(line_vector);
    }

    for(uint j = 0; j < fids.at(0).size(); ++j)
    {
        for(uint i = 0; i < fids.size(); ++i)
        {
            m_cls.fids.push_back(fids.at(i).at(j));
        }
    }

    m_cls.fidsSize.height = fids.size();
    m_cls.fidsSize.width = fids.at(0).size();

    std::vector< std::vector<float> > thrs;
    std::vector<float> line_float;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.thrs"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_float.clear();
        for(auto i : token)
        {
            line_float.push_back(boost::lexical_cast<float>(i));
        }
        thrs.push_back(line_float);
    }

    for(uint j = 0; j < thrs.at(0).size(); ++j)
    {
        for(uint i = 0; i < thrs.size(); ++i)
        {
            m_cls.thrs.push_back(thrs.at(i).at(j));
        }
    }

    std::vector< std::vector<int> > child;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.child"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_vector.clear();
        for(auto i : token)
        {
            line_vector.push_back(boost::lexical_cast<int>(i));
        }
        child.push_back(line_vector);
    }

    for(uint j = 0; j < child.at(0).size(); ++j)
    {
        for(uint i = 0; i < child.size(); ++i)
        {
            m_cls.child.push_back(child.at(i).at(j));
        }
    }


    std::vector< std::vector<float> > hs;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.hs"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_float.clear();
        for(auto i : token)
        {
            line_float.push_back(boost::lexical_cast<float>(i));
        }
        hs.push_back(line_float);
    }

    for(uint j = 0; j < hs.at(0).size(); ++j)
    {
        for(uint i = 0; i < hs.size(); ++i)
        {
            m_cls.hs.push_back(hs.at(i).at(j));
        }
    }

    std::vector< std::vector<float> > weights;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.weights"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_float.clear();
        for(auto i : token)
        {
            line_float.push_back(boost::lexical_cast<float>(i));
        }
        weights.push_back(line_float);
    }

    for(uint j = 0; j < weights.at(0).size(); ++j)
    {
        for(uint i = 0; i < weights.size(); ++i)
        {
            m_cls.weights.push_back(weights.at(i).at(j));
        }
    }

    std::vector< std::vector<int> > depth;
    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.depth"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        line_vector.clear();
        for(auto i : token)
        {
            line_vector.push_back(boost::lexical_cast<int>(i));
        }
        depth.push_back(line_vector);
    }

    for(uint j = 0; j < depth.at(0).size(); ++j)
    {
        for(uint i = 0; i < depth.size(); ++i)
        {
            m_cls.depth.push_back(depth.at(i).at(j));
        }
    }

    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.errs"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        for(auto i : token)
        {
            m_cls.errs.push_back(boost::lexical_cast<float>(i));
        }
    }

    BOOST_FOREACH(ptree::value_type &v, pt.get_child("classifier.clf.losses"))
    {
        line = v.second.data();
        const boost::tokenizer< boost::char_separator<char> > &token = parsingLine(line);
        for(auto i : token)
        {
            m_cls.losses.push_back(boost::lexical_cast<float>(i));
        }
    }

    m_cls.treeDepth = pt.get<int>("classifier.clf.treedepth");
}


boost::tokenizer< boost::char_separator<char> > Classifier::parsingLine(const std::string &_line)
{

    boost::char_separator<char> sep(" ");
    boost::tokenizer< boost::char_separator<char> > tok(_line, sep);
    return tok;
}
