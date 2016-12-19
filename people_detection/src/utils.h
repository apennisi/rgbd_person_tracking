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


#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <typeinfo>
#include "sse.hpp"
#include "structs.h"
#include "cutils.h"

namespace fpdw
{
    namespace utils
    {
        class Convolution
        {
            public:
                Convolution() {;}
                static void convTri1(cv::Mat &_img, const int &_smooth, cv::Mat &_output);
                static void convTri(cv::Mat &_img, const int &_r, cv::Mat &_output);
            private:
                static void convTri1Y( float *I, float *O, int h, float p, int s );
                static void convTriY( float *I, float *O, int h, int r, int s );
                static void convTri( float *I, float *O, int h, int w, int d, int r, int s );
                static void convConst(float *I, float *O, int &r, int &s, int &d, cv::Size &sz);
        };

        class RgbConvertion
        {
        public:
            static void process(const cv::Mat &_img, const structs::ColorSpace &_color, cv::Mat &_output);
        private:
            static void rgb2luv_setup( float z, float *mr, float *mg, float *mb,
                                  float &minu, float &minv, float &un, float &vn, float *lTable);
            static void rgb2luv(float *I, float *J, int n, float nrm );
            static void rgb2luv_sse(float *I, float *J, int n, float nrm );
            static void normalize(float *I, float *J, int n, float nrm );
        };

        class GradientMag
        {
            #define PI 3.14159265f
            public:
                GradientMag();
                void gradMag(const cv::Mat &_img, fpdw::structs::GradMag p, const int &_out);
                cv::Mat gradMagNorm(const cv::Mat &_img, const cv::Mat &_s, const fpdw::structs::GradMag &_p);
                cv::Mat gradientHist(const cv::Mat &_img, const cv::Mat &_s, const int &_binSize,
                                     const fpdw::structs::GradHist &_p, const bool &_full);
                cv::Mat M()
                {
                    return magnitude_gradient;
                }
                cv::Mat O()
                {
                    return approx_magitude;
                }

            private:
                static void gradHist( float *M, float *O, float *H, int h, int w,
                  int bin, int nOrients, int softBin, bool full );
                static void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full, float *acost);
                static void gradMagNorm( float *M, float *S, int h, int w, float norm );
                static void grad1( float *I, float *Gx, float *Gy, int h, int w, int x );
                static void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
                  int nb, int n, float norm, int nOrients, bool full, bool interpolate );
                static float *hogNormMatrix( float *H, int nOrients, int hb, int wb, int bin );
                static void hogChannels( float *H, const float *R, const float *N,
                  int hb, int wb, int nOrients, float clip, int type );
                static void hog( float *M, float *O, float *H, int h, int w, int binSize,
                  int nOrients, int softBin, bool full, float clip );
                static void fhog( float *M, float *O, float *H, int h, int w, int binSize,
                  int nOrients, int softBin, float clip );
            private:
                cv::Mat magnitude_gradient;
                cv::Mat approx_magitude;
                float *acost;
        };

        template <class T>
        class ImPad
        {
            public:
                static void impad(const cv::Mat &_input, const std::vector<int> &_pad,
                                  const structs::PadWith &_type, cv::Mat &_output, const float &_shrink)
                {
                    T *I, *O;
                    cv::Mat input = cvMatToMat(_input);
                    const int &k = static_cast<int>(_pad.size());
                    const uint &flag = (_type == structs::PadWith::FPDW_REPLICATE) ? 1 : 0;
                    const T &val = (_type == structs::PadWith::FPDW_REPLICATE|| _type == structs::PadWith::FPDW_NONE ) ? T(0) : T(1);
                    int pt, pb, pl, pr;
                    I = input.ptr<float>(0);

                    switch (k) {
                    case 1:
                        pt = pb = pl = pr = _pad[0] * _shrink;
                        break;
                    case 2:
                        pt = pb = _pad[0] * _shrink;
                        pl = pr = _pad[1] * _shrink;
                        break;
                    case 4:
                        pt = _pad[0] * _shrink;
                        pb = _pad[1] * _shrink;
                        pl = _pad[2] * _shrink;
                        pr = _pad[3] * _shrink;
                        break;
                    default:
                        std::cerr << "Input pad mush have 1, 2, or 4 values" << std::endl;
                        exit(-1);
                        break;
                    }

                    cv::Size out_size;
                    out_size.width = _input.cols + pl + pr;
                    out_size.height = _input.rows + pt + pb;
                    if( out_size.height < 0 || _input.rows <= -pt || _input.rows <= -pb ) out_size.height = 0;
                    if( out_size.width < 0 || _input.cols <= -pl || _input.cols <= -pr ) out_size.width = 0;

                    O = (T*) calloc (out_size.area()*_input.channels(), sizeof(T));

                    imPad(I, O, _input.rows, _input.cols, _input.channels(), pt, pb, pl, pr, flag, T(val));

                    switch (_input.channels()) {
                    case 1:
                        _output = matToCvMat1x(O, out_size);
                        break;
                    case 3:
                        _output = matToCvMat3x(O, out_size);
                        break;
                    default:
                        _output = matToCvMat6x(O, out_size);
                        break;
                    }

                    free(O);
                }

            private:
                static void imPad( T *A, T *B, int h, int w, int d, int pt, int pb,
                  int pl, int pr, int flag, T val )
                {
                    int h1=h+pt, hb=h1+pb, w1=w+pl, wb=w1+pr, x, y, z, mPad;
                    int ct=0, cb=0, cl=0, cr=0;
                    if(pt<0) { ct=-pt; pt=0; } if(pb<0) { h1+=pb; cb=-pb; pb=0; }
                    if(pl<0) { cl=-pl; pl=0; } if(pr<0) { w1+=pr; cr=-pr; pr=0; }
                    int *xs, *ys; x=pr>pl?pr:pl; y=pt>pb?pt:pb; mPad=x>y?x:y;
                    bool useLookup = ((flag==2 || flag==3) && (mPad>h || mPad>w))
                            || (flag==3 && (ct || cb || cl || cr ));
                    // helper macro for padding
#define PAD(XL,XM,XR,YT,YM,YB) \
                    for(x=0;  x<pl; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XL+cl)*h+YT+ct]; \
                    for(x=0;  x<pl; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XL+cl)*h+YM+ct]; \
                    for(x=0;  x<pl; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XL+cl)*h+YB-cb]; \
                    for(x=pl; x<w1; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XM+cl)*h+YT+ct]; \
                    for(x=pl; x<w1; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XM+cl)*h+YB-cb]; \
                    for(x=w1; x<wb; x++) for(y=0;  y<pt; y++) B[x*hb+y]=A[(XR-cr)*h+YT+ct]; \
                    for(x=w1; x<wb; x++) for(y=pt; y<h1; y++) B[x*hb+y]=A[(XR-cr)*h+YM+ct]; \
                    for(x=w1; x<wb; x++) for(y=h1; y<hb; y++) B[x*hb+y]=A[(XR-cr)*h+YB-cb];
                    // build lookup table for xs and ys if necessary
                    if( useLookup ) {
                        xs = (int*) wrMalloc(wb*sizeof(int)); int h2=(pt+1)*2*h;
                        ys = (int*) wrMalloc(hb*sizeof(int)); int w2=(pl+1)*2*w;
                        if( flag==2 ) {
                            for(x=0; x<wb; x++) { z=(x-pl+w2)%(w*2); xs[x]=z<w ? z : w*2-z-1; }
                            for(y=0; y<hb; y++) { z=(y-pt+h2)%(h*2); ys[y]=z<h ? z : h*2-z-1; }
                        } else if( flag==3 ) {
                            for(x=0; x<wb; x++) xs[x]=(x-pl+w2)%w;
                            for(y=0; y<hb; y++) ys[y]=(y-pt+h2)%h;
                        }
                    }
                    // pad by appropriate value
                    for( z=0; z<d; z++ ) {
                        // copy over A to relevant region in B
                        for( x=0; x<w-cr-cl; x++ )
                            memcpy(B+(x+pl)*hb+pt,A+(x+cl)*h+ct,sizeof(T)*(h-ct-cb));
                        // set boundaries of B to appropriate values
                        if( flag==0 && val!=0 ) { // "constant"
                            for(x=0;  x<pl; x++) for(y=0;  y<hb; y++) B[x*hb+y]=val;
                            for(x=pl; x<w1; x++) for(y=0;  y<pt; y++) B[x*hb+y]=val;
                            for(x=pl; x<w1; x++) for(y=h1; y<hb; y++) B[x*hb+y]=val;
                            for(x=w1; x<wb; x++) for(y=0;  y<hb; y++) B[x*hb+y]=val;
                        } else if( useLookup ) { // "lookup"
                            PAD( xs[x], xs[x], xs[x], ys[y], ys[y], ys[y] );
                        } else if( flag==1 ) {  // "replicate"
                            PAD( 0, x-pl, w-1, 0, y-pt, h-1 );
                        } else if( flag==2 ) { // "symmetric"
                            PAD( pl-x-1, x-pl, w+w1-1-x, pt-y-1, y-pt, h+h1-1-y );
                        } else if( flag==3 ) { // "circular"
                            PAD( x-pl+w, x-pl, x-pl-w, y-pt+h, y-pt, y-pt-h );
                        }
                        A += h*w;  B += hb*wb;
                    }
                    if( useLookup ) { wrFree(xs); wrFree(ys); }
#undef PAD
                }

        };


        template <class T>
        class ImResample
        {
            public:
                //ImResample() {;}
                static void resample(const cv::Mat &_img, const cv::Size &_sz, const float &_ratio, const int &_nChannels, cv::Mat &_output)
                {
                    const cv::Size &nDims = _img.size();
                    const int &nChannels = _nChannels;
                    const float &nrm = _ratio;
                    T *M, *O;

                    cv::Mat img;
                    if(nChannels == 1)
                    {
                        img = _img.t();
                        M = img.ptr<T>(0);
                    }
                    else
                    {
                        img = cvMatToMat(_img);
                        M = img.ptr<T>(0);

                    }

                    O = (T*) calloc(_sz.area()*nChannels, sizeof(T));
                    resample(M, O, nDims.height, _sz.height, nDims.width, _sz.width, nChannels, nrm);

                    if(nChannels == 1)
                    {
                        _output = matToCvMat1x(O, _sz);
                    }
                    else if(nChannels == 3)
                    {
                        _output = matToCvMat3x(O, _sz);
                    }
                    else
                    {
                        _output = matToCvMat6x(O, _sz);
                    }
                    free(O);
                }

            private:
                static void resample( T *A, T *B, int ha, int hb, int wa, int wb, int d, T r )
                {
                    int hn, wn, x, x1, y, z, xa, xb, ya; T *A0, *A1, *A2, *A3, *B0, wt, wt1;
                    T *C = (T*) alMalloc((ha+4)*sizeof(T),16); for(y=ha; y<ha+4; y++) C[y]=0;
                    bool sse = (typeid(T)==typeid(float)) && !(size_t(A)&15) && !(size_t(B)&15);
                    // get coefficients for resampling along w and h
                    int *xas, *xbs, *yas, *ybs; T *xwts, *ywts; int xbd[2], ybd[2];
                    resampleCoef( wa, wb, wn, xas, xbs, xwts, xbd, 0 );
                    resampleCoef( ha, hb, hn, yas, ybs, ywts, ybd, 4 );
                    if( wa==2*wb ) r/=2; if( wa==3*wb ) r/=3; if( wa==4*wb ) r/=4;
                    r/=T(1+1e-6); for( y=0; y<hn; y++ ) ywts[y] *= r;
                    // resample each channel in turn
                    for( z=0; z<d; z++ ) for( x=0; x<wb; x++ ) {
                        if(x==0) x1=0; xa=xas[x1]; xb=xbs[x1]; wt=xwts[x1]; wt1=1-wt; y=0;
                        A0=A+z*ha*wa+xa*ha; A1=A0+ha, A2=A1+ha, A3=A2+ha; B0=B+z*hb*wb+xb*hb;
                        // variables for SSE (simple casts to float)
                        float *Af0, *Af1, *Af2, *Af3, *Bf0, *Cf, *ywtsf, wtf, wt1f;
                        Af0=(float*) A0; Af1=(float*) A1; Af2=(float*) A2; Af3=(float*) A3;
                        Bf0=(float*) B0; Cf=(float*) C;
                        ywtsf=(float*) ywts; wtf=(float) wt; wt1f=(float) wt1;
                        // resample along x direction (A -> C)
                #define FORs(X) if(sse) for(; y<ha-4; y+=4) STR(Cf[y],X);
                #define FORr(X) for(; y<ha; y++) C[y] = X;
                        if( wa==2*wb ) {
                            FORs( ADD(LDu(Af0[y]),LDu(Af1[y])) );
                            FORr( A0[y]+A1[y] ); x1+=2;
                        } else if( wa==3*wb ) {
                            FORs( ADD(LDu(Af0[y]),LDu(Af1[y]),LDu(Af2[y])) );
                            FORr( A0[y]+A1[y]+A2[y] ); x1+=3;
                        } else if( wa==4*wb ) {
                            FORs( ADD(LDu(Af0[y]),LDu(Af1[y]),LDu(Af2[y]),LDu(Af3[y])) );
                            FORr( A0[y]+A1[y]+A2[y]+A3[y] ); x1+=4;
                        } else if( wa>wb ) {
                            int m=1; while( x1+m<wn && xb==xbs[x1+m] ) m++; float wtsf[4];
                            for( int x0=0; x0<(m<4?m:4); x0++ ) wtsf[x0]=float(xwts[x1+x0]);
                #define U(x) MUL( LDu(*(Af ## x + y)), SET(wtsf[x]) )
                #define V(x) *(A ## x + y) * xwts[x1+x]
                            if(m==1) { FORs(U(0));                     FORr(V(0)); }
                            if(m==2) { FORs(ADD(U(0),U(1)));           FORr(V(0)+V(1)); }
                            if(m==3) { FORs(ADD(U(0),U(1),U(2)));      FORr(V(0)+V(1)+V(2)); }
                            if(m>=4) { FORs(ADD(U(0),U(1),U(2),U(3))); FORr(V(0)+V(1)+V(2)+V(3)); }
                #undef U
                #undef V
                            for( int x0=4; x0<m; x0++ ) {
                                A1=A0+x0*ha; wt1=xwts[x1+x0]; Af1=(float*) A1; wt1f=float(wt1); y=0;
                                FORs(ADD(LD(Cf[y]),MUL(LDu(Af1[y]),SET(wt1f)))); FORr(C[y]+A1[y]*wt1);
                            }
                            x1+=m;
                        } else {
                            bool xBd = x<xbd[0] || x>=wb-xbd[1]; x1++;
                            if(xBd) memcpy(C,A0,ha*sizeof(T));
                            if(!xBd) FORs(ADD(MUL(LDu(Af0[y]),SET(wtf)),MUL(LDu(Af1[y]),SET(wt1f))));
                            if(!xBd) FORr( A0[y]*wt + A1[y]*wt1 );
                        }
                #undef FORs
                #undef FORr
                        // resample along y direction (B -> C)
                        if( ha==hb*2 ) {
                            T r2 = r/2; int k=((~((size_t) B0) + 1) & 15)/4; y=0;
                            for( ; y<k; y++ )  B0[y]=(C[2*y]+C[2*y+1])*r2;
                            if(sse) for(; y<hb-4; y+=4) STR(Bf0[y],MUL((float)r2,_mm_shuffle_ps(ADD(
                                                                                                    LDu(Cf[2*y]),LDu(Cf[2*y+1])),ADD(LDu(Cf[2*y+4]),LDu(Cf[2*y+5])),136)));
                            for( ; y<hb; y++ ) B0[y]=(C[2*y]+C[2*y+1])*r2;
                        } else if( ha==hb*3 ) {
                            for(y=0; y<hb; y++) B0[y]=(C[3*y]+C[3*y+1]+C[3*y+2])*(r/3);
                        } else if( ha==hb*4 ) {
                            for(y=0; y<hb; y++) B0[y]=(C[4*y]+C[4*y+1]+C[4*y+2]+C[4*y+3])*(r/4);
                        } else if( ha>hb ) {
                            y=0;
                            //if( sse && ybd[0]<=4 ) for(; y<hb; y++) // Requires SSE4
                            //  STR1(Bf0[y],_mm_dp_ps(LDu(Cf[yas[y*4]]),LDu(ywtsf[y*4]),0xF1));
                #define U(o) C[ya+o]*ywts[y*4+o]
                            if(ybd[0]==2) for(; y<hb; y++) { ya=yas[y*4]; B0[y]=U(0)+U(1); }
                            if(ybd[0]==3) for(; y<hb; y++) { ya=yas[y*4]; B0[y]=U(0)+U(1)+U(2); }
                            if(ybd[0]==4) for(; y<hb; y++) { ya=yas[y*4]; B0[y]=U(0)+U(1)+U(2)+U(3); }
                            if(ybd[0]>4)  for(; y<hn; y++) { B0[ybs[y]] += C[yas[y]] * ywts[y]; }
                #undef U
                        } else {
                            for(y=0; y<ybd[0]; y++) B0[y] = C[yas[y]]*ywts[y];
                            for(; y<hb-ybd[1]; y++) B0[y] = C[yas[y]]*ywts[y]+C[yas[y]+1]*(r-ywts[y]);
                            for(; y<hb; y++)        B0[y] = C[yas[y]]*ywts[y];
                        }
                    }
                    alFree(xas); alFree(xbs); alFree(xwts); alFree(C);
                    alFree(yas); alFree(ybs); alFree(ywts);
                }

                static void resampleCoef( int ha, int hb, int &n, int *&yas, int *&ybs, T *&wts, int bd[2], int pad=0 )
                {
                    const T s = T(hb)/T(ha), sInv = 1/s; T wt, wt0=T(1e-3)*s;
                    bool ds=ha>hb; int nMax; bd[0]=bd[1]=0;
                    if(ds) { n=0; nMax=ha+(pad>2 ? pad : 2)*hb; } else { n=nMax=hb; }
                    // initialize memory
                    wts = (T*)alMalloc(nMax*sizeof(T),16);
                    yas = (int*)alMalloc(nMax*sizeof(int),16);
                    ybs = (int*)alMalloc(nMax*sizeof(int),16);
                    if( ds ) for( int yb=0; yb<hb; yb++ ) {
                        // create coefficients for downsampling
                        T ya0f=yb*sInv, ya1f=ya0f+sInv, W=0;
                        int ya0=int(ceil(ya0f)), ya1=int(ya1f), n1=0;
                        for( int ya=ya0-1; ya<ya1+1; ya++ ) {
                            wt=s; if(ya==ya0-1) wt=(ya0-ya0f)*s; else if(ya==ya1) wt=(ya1f-ya1)*s;
                            if(wt>wt0 && ya>=0) { ybs[n]=yb; yas[n]=ya; wts[n]=wt; n++; n1++; W+=wt; }
                        }
                        if(W>1) for( int i=0; i<n1; i++ ) wts[n-n1+i]/=W;
                        if(n1>bd[0]) bd[0]=n1;
                        while( n1<pad ) { ybs[n]=yb; yas[n]=yas[n-1]; wts[n]=0; n++; n1++; }
                    } else for( int yb=0; yb<hb; yb++ ) {
                        // create coefficients for upsampling
                        T yaf = (T(.5)+yb)*sInv-T(.5); int ya=(int) floor(yaf);
                        wt=1; if(ya>=0 && ya<ha-1) wt=1-(yaf-ya);
                        if(ya<0) { ya=0; bd[0]++; } if(ya>=ha-1) { ya=ha-1; bd[1]++; }
                        ybs[yb]=yb; yas[yb]=ya; wts[yb]=wt;
                    }
                }

        };
    }
}

#endif
