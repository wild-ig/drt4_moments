/* 
MIT License

Copyright (c) 2020 wild-ig 
*/
#include <vector>
#include <numeric>
#include <opencv2\opencv.hpp>
#include "moments.hpp"

using namespace std;
using namespace cv;

//power arrays
double *d1, *d2, *d3, *d4, *a3, *a4, *s4;

double product(const vector<long> &mat, double power[], int many)
{
    double sum = 0.0;
    for(int i = 0; i < many; i++)
        sum += static_cast<double>(mat[i]) * power[i];

    return sum;
}

void pre_compute_power_arrays(const Size s) {
    const int width = s.width;
    const int height = s.height;

    //power arrays
    d1 = new double [width + height];
    d2 = new double [width + height];
    d3 = new double [width + height];
    d4 = new double [width + height];
    a3 = new double [width + height];
    a4 = new double [width + height];
    s4 = new double [width + height * 2];

    for (int k = 0; k < width + height; ++k)
    {
        d1[k] = k;
        double k2 = static_cast<double>(k) * static_cast<double>(k);
        d2[k] = k2;
        d3[k] = k2 * static_cast<double>(k);
        d4[k] = k2 * k2;
        a3[k] = pow(static_cast<double>(k - width + 1), 3);
        a4[k] = pow(static_cast<double>(k - width + 1), 4);
    }

    for (int k = 0; k < width + height * 2; ++k)
    {
        s4[k] = pow(static_cast<double>(k), 4);
    }
}


Moments4 drt4_moments(const Mat& image)
{
    Size s = image.size();
    const int width = s.width;
    const int height = s.height;

    Moments4 m;

    // projection arrays
    vector<long> vert(width, 0);
    vector<long> horz(height, 0);
    vector<long> diag(width+height, 0);
    vector<long> anti(width+height, 0);
    vector<long> x_2y(width+height*2, 0);

    long* hptr = &horz[0];
    long* vptr = &vert[0];
    long* dptr = &diag[0];
    long* aptr = &anti[height - 1];
    long* x2yptr = &x_2y[0];

    for (int i = 0; i < height; i++)
    {
        const uchar* p = image.ptr<uchar>(i);

        for(int j = 0; j < width; j++)
        {
            vptr[j] += p[j];
            hptr[i] += p[j];
            dptr[j] += p[j];
            aptr[j] += p[j];
            x2yptr[j] += p[j];
        }

        x2yptr+=2;
        dptr++;
        aptr--;
    }

    m.m00 = accumulate(begin(vert), end(vert), 0.0);
    m.m10 = product(vert, d1, width);
    m.m01 = product(horz, d1, height);
    m.m20 = product(vert, d2, width);
    m.m02 = product(horz, d2, height);
    m.m30 = product(vert, d3, width);
    m.m03 = product(horz, d3, height);
    m.m40 = product(vert, d4, width);
    m.m04 = product(horz, d4, height);
    m.m11 = (product(diag, d2, width+height) - m.m02 - m.m20) / 2.0;
    double temp_1 = product(diag, d3, width+height) / 6.0;
    double temp_2 = product(anti, a3, width+height) / 6.0;
    m.m12 = temp_1 + temp_2 - m.m30/3.0;
    m.m21 = temp_1 - temp_2 - m.m03/3.0;

    // 4th order diagonal and anti-diagonal projection moments
    double md_4 = product(diag, d4, width+height);
    double ma_4 = product(anti, a4, width+height);

    m.m22 = (md_4 + ma_4 - 2*m.m40 - 2*m.m04) / 12.0;

    // 4th order moment along x+2y projection
    double ms_4 = product(x_2y, s4, width+height*2);

    m.m13 = (ms_4 - 2*md_4 + m.m40 - 14.0 * m.m04 - 12.0  * m.m22) / 24.0;
    m.m31 = (ms_4 - m.m40 - 24 * m.m22 - 32*m.m13 - 16*m.m04) / 8.0;

    return m;
}

Moments4 opencv4_moments(const Mat& image)
{
    Size s = image.size();
    Moments4 m;

    for(int y = 0; y < s.height; y++ )
    {
        const uchar* p = image.ptr<uchar>(y);
        long x0 = 0;
        double x1 = 0.0, x2 = 0.0, x3 = 0.0, x4 = 0.0;

        for(int x = 0; x < s.width; x++ ) {
            double xp = x * p[x], xxp = x * xp, xxxp = xxp * x;

            x0 += p[x];
            x1 += xp;
            x2 += xxp;
            x3 += xxxp;
            x4 += xxxp * x;
        }

        double py = y * x0, sy = y * y, cy = sy * y;

        m.m04 += py * cy;
        m.m13 += x1 * cy;
        m.m31 += x3 * y;
        m.m40 += x4;
        m.m22 += x2 * sy;
        m.m03 += py * sy;  
        m.m12 += x1 * sy;  
        m.m21 += x2 * y;
        m.m30 += x3;
        m.m02 += x0 * sy;
        m.m11 += x1 * y;
        m.m20 += x2;
        m.m01 += py;
        m.m10 += x1;
        m.m00 += x0;
    }

    return m;
}

