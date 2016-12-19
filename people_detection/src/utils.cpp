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


#include "utils.h"

using namespace fpdw::utils;

void Convolution::convTri1Y(float *I, float *O, int h, float p, int s )
{
#define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
    int j=0, k=((~((size_t) O) + 1) & 15)/4, h2=(h-1)/2;
    if( s==2 )
    {
        for( ; j<k; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        for( ; j<h2-4; j+=4 ) STR(O[j],_mm_shuffle_ps(C4(2,1),C4(2,5),136));
        for( ; j<h2; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        if( h%2==0 ) O[j]=I[2*j]+(1+p)*I[2*j+1];
    } else {
        O[j]=(1+p)*I[j]+I[j+1]; j++; if(k==0) k=(h<=4) ? h-1 : 4;
        for( ; j<k; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        for( ; j<h-4; j+=4 ) STR(O[j],C4(1,0));
        for( ; j<h-1; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        O[j]=I[j-1]+(1+p)*I[j];
    }
#undef C4
}

void Convolution::convTriY(float *I, float *O, int h, int r, int s)
{
    r++; float t, u; int j, r0=r-1, r1=r+1, r2=2*h-r, h0=r+1, h1=h-r+1, h2=h;
    u=t=I[0]; for( j=1; j<r; j++ ) u+=t+=I[j]; u=2*u-t; t=0;
    if( s==1 ) {
        O[0]=u; j=1;
        for(; j<h0; j++) O[j] = u += t += I[r-j]  + I[r0+j] - 2*I[j-1];
        for(; j<h1; j++) O[j] = u += t += I[j-r1] + I[r0+j] - 2*I[j-1];
        for(; j<h2; j++) O[j] = u += t += I[j-r1] + I[r2-j] - 2*I[j-1];
    } else {
        int k=(s-1)/2; h2=(h/s)*s; if(h0>h2) h0=h2; if(h1>h2) h1=h2;
        if(++k==s) { k=0; *O++=u; } j=1;
        for(;j<h0;j++) { u+=t+=I[r-j] +I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h1;j++) { u+=t+=I[j-r1]+I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h2;j++) { u+=t+=I[j-r1]+I[r2-j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
    }
}

void Convolution::convTri(float *I, float *O, int h, int w, int d, int r, int s)
{
    r++; float nrm = 1.0f/(r*r*r*r); int i, j, k=(s-1)/2, h0, h1, w0;
    if(h%4==0) h0=h1=h; else { h0=h-(h%4); h1=h0+4; } w0=(w/s)*s;
    float *T=(float*) alMalloc(2*h1*sizeof(float),16), *U=T+h1;
    while(d-- > 0) {
        // initialize T and U
        for(j=0; j<h0; j+=4) STR(U[j], STR(T[j], LDu(I[j])));
        for(i=1; i<r; i++) for(j=0; j<h0; j+=4) INC(U[j],INC(T[j],LDu(I[j+i*h])));
        for(j=0; j<h0; j+=4) STR(U[j],MUL(nrm,(SUB(MUL(2,LD(U[j])),LD(T[j])))));
        for(j=0; j<h0; j+=4) STR(T[j],0);
        for(j=h0; j<h; j++ ) U[j]=T[j]=I[j];
        for(i=1; i<r; i++) for(j=h0; j<h; j++ ) U[j]+=T[j]+=I[j+i*h];
        for(j=h0; j<h; j++ ) { U[j] = nrm * (2*U[j]-T[j]); T[j]=0; }
        // prepare and convolve each column in turn
        k++; if(k==s) { k=0; convTriY(U,O,h,r-1,s); O+=h/s; }
        for( i=1; i<w0; i++ )
        {
            float *Il=I+(i-1-r)*h;
            if(i<=r)
                Il=I+(r-i)*h;
            float *Im=I+(i-1)*h;
            float *Ir=I+(i-1+r)*h;
            if(i>w-r)
                Ir=I+(2*w-r-i)*h;

            for( j=0; j<h0; j+=4 )
            {
                INC(T[j],ADD(LDu(Il[j]),LDu(Ir[j]),MUL(-2,LDu(Im[j]))));
                INC(U[j],MUL(nrm,LD(T[j])));
            }
            for( j=h0; j<h; j++ )
                U[j]+=nrm*(T[j]+=Il[j]+Ir[j]-2*Im[j]);
            k++;
            if(k==s)
            {
                k=0; convTriY(U,O,h,r-1,s);
                O+=h/s;
            }
        }
        I+=w*h;
    }
    alFree(T);
}

void Convolution::convConst(float *I, float *O, int &r, int &s, int &d, cv::Size &sz)
{
    int h = sz.height;
    int w = sz.width;

    float p = (float)(r);

    const float nrm = 1.0f/((p+2)*(p+2));
    int i, j, h0 = h-(h%4);
    float *Il, *Im, *Ir;
    float *T =  (float*)(alMalloc(h*sizeof(float), 16));
    int h_s = h/s;

    for( int d0=0; d0<d; d0++ )
    {
        for( i=s/2; i<w; i+=s )
        {
            Il = Im = Ir = I + i * h + d0*h*w;
            if(i>0)
            {
                Il-=h;
            }

            if(i<w-1)
            {
                Ir+=h;
            }

            for( j=0; j<h0; j+=4 )
            {
                STR(T[j],MUL(nrm,ADD(ADD(LDu(Il[j]),MUL(p,LDu(Im[j]))),LDu(Ir[j]))));
            }

            for( j=h0; j<h; j++ )
            {
                T[j]=nrm*(Il[j]+p*Im[j]+Ir[j]);
            }

            convTri1Y(T, O, h, p, s);
            O+=h_s;
        }

    }

    alFree(T);
}


void Convolution::convTri1(cv::Mat &_img, const int &_smooth, cv::Mat &_output)
{
    const int smooth = _smooth;

    int r;
    if(smooth > 0 && smooth <= 1)
    {
        r = (int)(12/_smooth/(_smooth + 2) - 2);
    }
    else
    {
        r = smooth;
    }
    cv::Mat float_image = cvMatToMat(_img).clone();

    int s = 1;
    int d = _img.channels();
    cv::Size sz = _img.size();

    float *O = (float*) calloc(sz.width*sz.height*d, sizeof(float));
    convConst(float_image.ptr<float>(0), O, r, s, d, sz);

    if(d == 1)
    {
        _output = matToCvMat1x(O, _img.size()).clone();
    }
    else if(d == 3)
    {
        _output = matToCvMat3x(O, _img.size()).clone();
    }
    else
    {
        _output = matToCvMat6x(O, _img.size()).clone();
    }

    free(O);
}

void Convolution::convTri(cv::Mat &_img, const int &_r, cv::Mat &_output)
{
    const int r = _r;
    int s = 1;
    int d = _img.channels();
    cv::Size sz = _img.size();
    cv::Mat float_image = _img.t();
    float *O = (float*) calloc(sz.width*sz.height*d, sizeof(float));
    convTri(float_image.ptr<float>(0), O, sz.height, sz.width, d, r, s);

    if(d == 1)
    {
        _output = matToCvMat1x(O, _img.size()).clone();
    }
    else if(d == 3)
    {
        _output = matToCvMat3x(O, _img.size()).clone();
    }
    else
    {
        _output = matToCvMat6x(O, _img.size()).clone();
    }

    free(O);
}

void RgbConvertion::process(const cv::Mat &_img, const structs::ColorSpace &_color, cv::Mat &_output)
{
    cv::Mat float_image = _img.clone();
    cv::cvtColor(float_image, float_image, CV_BGR2RGB);
    float_image.convertTo(float_image, CV_32FC3);

    float_image = cvMatToMat(float_image).clone();

    float *ptr = float_image.ptr<float>(0);

    float *J=nullptr;
    if(_color == structs::ColorSpace::FPDW_ORIG)
    {
        if(_img.type() == CV_32FC3)
        {
            _output = _img.clone();
        }
        else
        {
            J = (float*) calloc((_img.size().area() * _img.channels()), sizeof(float));
            normalize(ptr, J, _img.size().area(), 1./255.);
        }
    }
    else if(_color == structs::ColorSpace::FPDW_LUV)
    {
        J = (float*) calloc((_img.size().area() * _img.channels()), sizeof(float));
        int d = _img.channels();
        for(unsigned i=0; i<d/3; i++)
        {
            rgb2luv_sse(ptr+i*_img.size().area()*3,(float*)(J+i*_img.size().area()*3), _img.size().area(), 1./255.);
        }
    }

    _output = matToCvMat3x(J, _img.size());

    if(J)
    {free(J);}
}

void RgbConvertion::rgb2luv_setup(float z, float *mr, float *mg, float *mb, float &minu, float &minv, float &un, float &vn, float *lTable)
{
    //METTERE OPENMP
    const float y0=(float) ((6.0/29)*(6.0/29)*(6.0/29));
    const float a= (float) ((29.0/3)*(29.0/3)*(29.0/3));
    un=(float) 0.197833; vn=(float) 0.468331;
    mr[0]=(float) 0.430574*z; mr[1]=(float) 0.222015*z; mr[2]=(float) 0.020183*z;
    mg[0]=(float) 0.341550*z; mg[1]=(float) 0.706655*z; mg[2]=(float) 0.129553*z;
    mb[0]=(float) 0.178325*z; mb[1]=(float) 0.071330*z; mb[2]=(float) 0.939180*z;
    float maxi=(float) 1.0/270; minu=-88*maxi; minv=-134*maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    float y, l;
    //#pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) shared(lTable) private(y, l)
    for(int i=0; i<1025; i++)
    {
        y = (float) (i/1024.0);
        l = y>y0 ? 116*(float)pow((double)y,1.0/3.0)-16 : y*a;
        lTable[i] = l*maxi;
    }

    //#pragma omp parallel for num_threads( omp_get_num_procs() * omp_get_num_threads() ) shared(lTable)
    for(int i=1025; i<1064; i++)
    {
        lTable[i]=lTable[i-1];
    }

}

void RgbConvertion::rgb2luv(float *I, float *J, int n, float nrm)
{
    float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float lTable[1064];
    rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn, lTable);
    float *L=J, *U=L+n, *V=U+n; float *R=I, *G=R+n, *B=G+n;
    for( int i=0; i<n; i++ )
    {
        float r, g, b, x, y, z, l;
        r=(float)*R++; g=(float)*G++; b=(float)*B++;
        x = mr[0]*r + mg[0]*g + mb[0]*b;
        y = mr[1]*r + mg[1]*g + mb[1]*b;
        z = mr[2]*r + mg[2]*g + mb[2]*b;
        l = lTable[(int)(y*1024)];
        *(L++) = l;
        z = 1/(x + 15*y + 3*z + (float)1e-35);
        *(U++) = l * (13*4*x*z - 13*un) - minu;
        *(V++) = l * (13*9*y*z - 13*vn) - minv;
    }
}


void RgbConvertion::rgb2luv_sse(float *I, float *J, int n, float nrm)
{
    const int k=256;
    float R[k], G[k], B[k];
    if( (size_t(R)&15||size_t(G)&15||size_t(B)&15||size_t(I)&15||size_t(J)&15)|| n%4>0 )
    {
        rgb2luv(I,J,n,nrm);
        return;
    }
    int i=0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];;

    float lTable[1064];
    rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn, lTable);

    while( i<n )
    {
        n1 = i+k;
        if(n1>n) n1=n;
        float *J1=J+i;
        float *R1, *G1, *B1;
        // convert to floats (and load input into cache)
        if( typeid(float) != typeid(float) )
        {
            R1=R; G1=G; B1=B; float *Ri=I+i, *Gi=Ri+n, *Bi=Gi+n;
            for( i1=0; i1<(n1-i); i1++ )
            {
                R1[i1] = (float) *Ri++;
                G1[i1] = (float) *Gi++;
                B1[i1] = (float) *Bi++;
            }
        }
        else
        {
            R1=((float*)I)+i; G1=R1+n; B1=G1+n;
        }
        // compute RGB -> XYZ
        for( int j=0; j<3; j++ )
        {
            __m128 _mr, _mg, _mb, *_J=(__m128*) (J1+j*n);
            __m128 *_R=(__m128*) R1, *_G=(__m128*) G1, *_B=(__m128*) B1;
            _mr=SET(mr[j]); _mg=SET(mg[j]); _mb=SET(mb[j]);

            for( i1=i; i1<n1; i1+=4 )
            {
                *(_J++) = ADD( ADD(MUL(*(_R++),_mr), MUL(*(_G++),_mg)),MUL(*(_B++),_mb));
            }
        }
        { // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15=SET(15.0f); _c3=SET(3.0f); _cEps=SET(1e-35f);
            _c52=SET(52.0f); _c117=SET(117.0f), _c1024=SET(1024.0f);
            _cun=SET(13*un); _cvn=SET(13*vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X=(__m128*) J1; _Y=(__m128*) (J1+n); _Z=(__m128*) (J1+2*n);
            for( i1=i; i1<n1; i1+=4 )
            {
                _x = *_X; _y=*_Y; _z=*_Z;
                _z = RCP(ADD(_x,ADD(_cEps,ADD(MUL(_c15,_y),MUL(_c3,_z)))));
                *(_X++) = MUL(_c1024,_y);
                *(_Y++) = SUB(MUL(MUL(_c52,_x),_z),_cun);
                *(_Z++) = SUB(MUL(MUL(_c117,_y),_z),_cvn);
            }
        }
        { // perform lookup for L and finalize computation of U and V
            for( i1=i; i1<n1; i1++ )
            {
                J[i1] = lTable[(int)J[i1]];
            }
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L=(__m128*) J1; _U=(__m128*) (J1+n); _V=(__m128*) (J1+2*n);
            _cminu=SET(minu); _cminv=SET(minv);
            for( i1=i; i1<n1; i1+=4 )
            {
                _l = *(_L++);
                *_U = SUB(MUL(_l,*_U),_cminu); _U++;
                *_V = SUB(MUL(_l,*_V),_cminv); _V++;
            }
        }
        i = n1;
    }
}

void RgbConvertion::normalize(float *I, float *J, int n, float nrm)
{
    for(int i=0; i<n; ++i)
    {
        *(J++)=(float)*(I++)*nrm;
    }
}


GradientMag::GradientMag()
{
    const int n=10000, b=10; int i;
    static float a[n*2+b*2];
    acost=a+n+b;
    for( i=-n-b; i<-n; i++ )   acost[i]=CV_PI;
    for( i=-n; i<n; i++ )      acost[i]=float(acos(i/float(n)));
    for( i=n; i<n+b; i++ )     acost[i]=0;
    for( i=-n-b; i<n/10; i++ ) if( acost[i] > CV_PI-1e-6f ) acost[i]=CV_PI-1e-6f;
}

void GradientMag::gradMag(const cv::Mat &_img,
                          fpdw::structs::GradMag p, const int &_out)
{
    cv::Mat float_image = cvMatToMat(_img).clone();

    float *ptr = float_image.ptr<float>(0);
    int h = _img.rows;
    int w = _img.cols;
    int d = _img.channels();

    if(p.colorChn > 0 && p.colorChn <=d)
    {
        ptr += h*w*(p.colorChn-1);
        d = 1;
    }

    float *M, *O = 0;
    M = (float*) calloc(w*h, sizeof(float));
    if(_out == 2)
    {
        O = (float*) calloc(w*h, sizeof(float));
    }

    gradMag(ptr, M, O, h, w, d, int(p.full) > 0, acost);

    magnitude_gradient = matToCvMat1x(M, _img.size());
    if(_out == 2) approx_magitude = matToCvMat1x(O, _img.size());

    free(M);
    free(O);
}

cv::Mat GradientMag::gradMagNorm(const cv::Mat &_img, const cv::Mat &_s, const fpdw::structs::GradMag &_p)
{
    int h = _img.rows;
    int w = _img.cols;
    int d = _img.channels();
    cv::Mat img = _img.t();
    cv::Mat nS = _s.t();
    float *M = img.ptr<float>(0);
    float *S = nS.ptr<float>(0);
    float norm = _p.normConst;
    gradMagNorm(M, S, h, w, norm);

    return matToCvMat1x(M, _img.size());
}

cv::Mat GradientMag::gradientHist(const cv::Mat &_img, const cv::Mat &_s, const int &_binSize,
                                  const fpdw::structs::GradHist &_p, const bool &_full)
{
    int h = _img.rows;
    int w = _img.cols;
    int d = _img.channels();
    cv::Mat img = _img.t();
    cv::Mat nS = _s.t();
    float *M = img.ptr<float>(0);
    float *S = nS.ptr<float>(0);
    int useHog = _p.useHog;
    float clipHog = _p.clipHog;
    int hb = h/_binSize;
    int wb = w/_binSize;
    int nChns = (useHog == 0) ? _p.nOrients : ( useHog == 1 ? _p.nOrients*4 : _p.nOrients*3+5 );
    if(_p.nOrients != 0)
    {
        float *H = (float *) calloc(wb*hb*nChns, sizeof(float));

        if(useHog == 0)
        {
            gradHist(M, S, H, h, w, _binSize, _p.nOrients, _p.softBin, _full);
        }
        else if(useHog == 1)
        {
            hog( M, S, H, h, w, _binSize, _p.nOrients, _p.softBin, _full, clipHog );
        }
        else
        {
            fhog( M, S, H, h, w, _binSize, _p.nOrients, _p.softBin, clipHog );
        }

        cv::Mat out = matToCvMat6x(H, cv::Size(wb,hb));

        free(H);

        return out;
    }
    else
    {
        return cv::Mat();
    }
}

void GradientMag::gradHist(float *M, float *O, float *H, int h, int w, int bin, int nOrients, int softBin, bool full)
{
    const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
    const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;
    O0=(int*)alMalloc(h*sizeof(int),16); M0=(float*) alMalloc(h*sizeof(float),16);
    O1=(int*)alMalloc(h*sizeof(int),16); M1=(float*) alMalloc(h*sizeof(float),16);
    // main loop
    for( x=0; x<w0; x++ ) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);

        if( softBin<0 && softBin%2==0 ) {
            // no interpolation w.r.t. either orienation or spatial bin
            H1=H+(x/bin)*hb;
#define GH H1[O0[y]]+=M0[y]; y++;
            if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH

        } else if( softBin%2==0 || bin==1 ) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1=H+(x/bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH

        } else {
            // interpolate using trilinear interpolation
            float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
            bool hasLf, hasRt; int xb0, yb0;
            if( x==0 ) { init=(0+.5f)*sInv-0.5f; xb=init; }
            hasLf = xb>=0; xb0 = hasLf?(int)xb:-1; hasRt = xb0 < wb-1;
            xd=xb-xb0; xb+=sInv; yb=init; y=0;
            // macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
    ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
            // leading rows, no top bin
            for( ; y<bin/2; y++ ) {
                yb0=-1; GHinit;
                if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
                if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
            }
            // main rows, has top and bottom bins, use SSE for minor speedup
            if( softBin<0 ) for( ; ; y++ ) {
                yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; _m0=SET(M0[y]);
                if(hasLf) { _m=SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); }
                if(hasRt) { _m=SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); }
            } else for( ; ; y++ ) {
                yb0 = (int) yb; if(yb0>=hb-1) break; GHinit;
                _m0=SET(M0[y]); _m1=SET(M1[y]);
                if(hasLf) { _m=SET(0,0,ms[1],ms[0]);
                    GH(H0+O0[y],_m,_m0); GH(H0+O1[y],_m,_m1); }
                if(hasRt) { _m=SET(0,0,ms[3],ms[2]);
                    GH(H0+O0[y]+hb,_m,_m0); GH(H0+O1[y]+hb,_m,_m1); }
            }
            // final rows, no bottom bin
            for( ; y<h0; y++ ) {
                yb0 = (int) yb; GHinit;
                if(hasLf) { H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; }
                if(hasRt) { H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; }
            }
#undef GHinit
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
        x=0; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        y=0; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
        y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    }
}

void GradientMag::gradMag(float *I, float *M, float *O, int h, int w, int d, bool full, float *acost)
{
    int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float acMult=10000.0f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    h4=(h%4==0) ? h : h-(h%4)+4; s=d*h4*sizeof(float);
    M2=(float*) alMalloc(s,16); _M2=(__m128*) M2;
    Gx=(float*) alMalloc(s,16); _Gx=(__m128*) Gx;
    Gy=(float*) alMalloc(s,16); _Gy=(__m128*) Gy;
    // compute gradient magnitude and orientation for each column
    for( x=0; x<w; x++ ) {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for(c=0; c<d; c++) {
            grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );
            for( y=0; y<h4/4; y++ ) {
                y1=h4/4*c+y;
                _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
                if( c==0 ) continue; _m = CMPGT( _M2[y1], _M2[y] );
                _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
                _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
                _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
            }
        }
        // compute gradient mangitude (M) and normalize Gx
        for( y=0; y<h4/4; y++ ) {
            _m = MIN( RCPSQRT(_M2[y]), SET(1e10f) );
            _M2[y] = RCP(_m);
            if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
            if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
        };
        memcpy( M+x*h, M2, h*sizeof(float) );
        // compute and store gradient orientation (O) via table lookup
        if( O!=0 ) for( y=0; y<h; y++ ) O[x*h+y] = acost[(int)Gx[y]];
        if( O!=0 && full ) {
            y1=((~size_t(O+x*h)+1)&15)/4; y=0;
            for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
            for( ; y<h-4; y+=4 ) STRu( O[y+x*h],
                    ADD( LDu(O[y+x*h]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET(PI)) ) );
            for( ; y<h; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
        }
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}

void GradientMag::gradMagNorm(float *M, float *S, int h, int w, float norm)
{
    __m128 *_M, *_S, _norm;
    int i=0, n=h*w, n4=n/4;
    _S = (__m128*) S;
    _M = (__m128*) M;
    _norm = SET(norm);
    bool sse = !(size_t(M)&15) && !(size_t(S)&15);
    if(sse)
        for(; i<n4; i++)
        {
            *_M=MUL(*_M,RCP(ADD(*_S++,_norm)));
            _M++;
        }
    if(sse)
        i*=4;
    for(; i<n; i++)
        M[i] /= (S[i] + norm);
}



void GradientMag::grad1(float *I, float *Gx, float *Gy, int h, int w, int x)
{
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip=I-h; In=I+h; r=.5f;
    if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
    if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
        for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
    } else {
        _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
        for(y=0; y<h; y+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip=I; In=Ip+1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1=((~((size_t) Gy) + 1) & 15)/4; if(y1==0) y1=4; if(y1>h-1) y1=h-1;
    GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
    _r = SET(.5f); _G=(__m128*) Gy;
    for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
        *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
    for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

void GradientMag::gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1, int nb, int n, float norm, int nOrients, bool full, bool interpolate)
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult=(float)nOrients/(full?2*PI:PI); const int oMax=nOrients*nb;
    const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
    const __m128i _oMax=SET(oMax), _nb=SET(nb);
    // perform the majority of the work with sse
    _O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
    if( interpolate ) for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(_o); _od=SUB(_o,CVT(_o0));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        _o1=ADD(_o0,_nb); _o1=AND(CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
        _m=MUL(LDu(M[i]),_norm); *_M1=MUL(_od,_m); *_M0++=SUB(_m,*_M1); _M1++;
    } else for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        *_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
    }
    // compute trailing locations without sse
    if( interpolate ) for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) o; od=o-o0;
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
        m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
    } else for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) (o+.5f);
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
    }
}

float *GradientMag::hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin)
{
    float *N, *N1, *n; int o, x, y, dx, dy, hb1=hb+1, wb1=wb+1;
    float eps = 1e-4f/4/bin/bin/bin/bin; // precise backward equality
    N = (float*) wrCalloc(hb1*wb1,sizeof(float)); N1=N+hb1+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) for( y=0; y<hb; y++ )
        N1[x*hb1+y] += H[o*wb*hb+x*hb+y]*H[o*wb*hb+x*hb+y];
    for( x=0; x<wb-1; x++ ) for( y=0; y<hb-1; y++ ) {
        n=N1+x*hb1+y; *n=1/float(sqrt(n[0]+n[1]+n[hb1]+n[hb1+1]+eps)); }
    x=0;     dx= 1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy= 0; for(y=0; y<hb1; y++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 0; for( y=0; y<hb1; y++) N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=0;     dx= 0; dy= 1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=hb1-1; dx= 0; dy=-1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    return N;
}

void GradientMag::hogChannels(float *H, const float *R, const float *N, int hb, int wb, int nOrients, float clip, int type)
{
#define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
    const float r=.2357f; int o, x, y, c; float t;
    const int nb=wb*hb, nbo=nOrients*nb, hb1=hb+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) {
        const float *R1=R+o*nb+x*hb, *N1=N+x*hb1+hb1+1;
        float *H1 = (type<=1) ? (H+o*nb+x*hb) : (H+x*hb);
        if( type==0) for( y=0; y<hb; y++ ) {
            // store each orientation and normalization (nOrients*4 channels)
            c=-1; GETT(0); H1[c*nbo+y]=t; GETT(1); H1[c*nbo+y]=t;
            GETT(hb1); H1[c*nbo+y]=t; GETT(hb1+1); H1[c*nbo+y]=t;
        } else if( type==1 ) for( y=0; y<hb; y++ ) {
            // sum across all normalizations (nOrients channels)
            c=-1; GETT(0); H1[y]+=t*.5f; GETT(1); H1[y]+=t*.5f;
            GETT(hb1); H1[y]+=t*.5f; GETT(hb1+1); H1[y]+=t*.5f;
        } else if( type==2 ) for( y=0; y<hb; y++ ) {
            // sum across all orientations (4 channels)
            c=-1; GETT(0); H1[c*nb+y]+=t*r; GETT(1); H1[c*nb+y]+=t*r;
            GETT(hb1); H1[c*nb+y]+=t*r; GETT(hb1+1); H1[c*nb+y]+=t*r;
        }
    }
#undef GETT
}

void GradientMag::hog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, bool full, float clip)
{
    float *N, *R; const int hb=h/binSize, wb=w/binSize, nb=hb*wb;
    // compute unnormalized gradient histograms
    R = (float*) wrCalloc(wb*hb*nOrients,sizeof(float));
    gradHist( M, O, R, h, w, binSize, nOrients, softBin, full );
    // compute block normalization values
    N = hogNormMatrix( R, nOrients, hb, wb, binSize );
    // perform four normalizations per spatial block
    hogChannels( H, R, N, hb, wb, nOrients, clip, 0 );
    wrFree(N); wrFree(R);
}

void GradientMag::fhog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, float clip)
{
    const int hb=h/binSize, wb=w/binSize, nb=hb*wb, nbo=nb*nOrients;
    float *N, *R1, *R2; int o, x;
    // compute unnormalized constrast sensitive histograms
    R1 = (float*) wrCalloc(wb*hb*nOrients*2,sizeof(float));
    gradHist( M, O, R1, h, w, binSize, nOrients*2, softBin, true );
    // compute unnormalized contrast insensitive histograms
    R2 = (float*) wrCalloc(wb*hb*nOrients,sizeof(float));
    for( o=0; o<nOrients; o++ ) for( x=0; x<nb; x++ )
        R2[o*nb+x] = R1[o*nb+x]+R1[(o+nOrients)*nb+x];
    // compute block normalization values
    N = hogNormMatrix( R2, nOrients, hb, wb, binSize );
    // normalized histograms and texture channels
    hogChannels( H+nbo*0, R1, N, hb, wb, nOrients*2, clip, 1 );
    hogChannels( H+nbo*2, R2, N, hb, wb, nOrients*1, clip, 1 );
    hogChannels( H+nbo*3, R1, N, hb, wb, nOrients*2, clip, 2 );
    wrFree(N); wrFree(R1); wrFree(R2);
}
