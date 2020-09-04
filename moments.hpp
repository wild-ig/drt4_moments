/* 
MIT License

Copyright (c) 2020 wild-ig 
*/
#ifndef MOMEMTS
#define MOMENTS

struct Moments4 {
    double m00 = 0.0;
    double m10 = 0.0, m20 = 0.0, m30 = 0.0, m40 = 0.0;
    double m01 = 0.0, m02 = 0.0, m03 = 0.0, m04 = 0.0;
    double m11 = 0.0, m21 = 0.0, m12 = 0.0;
    double m31 = 0.0, m13 = 0.0, m22 = 0.0;
};

void pre_compute_power_arrays(const cv::Size s);
Moments4 drt4_moments(const cv::Mat& image);
Moments4 opencv4_moments(const cv::Mat& image);

#endif // MOMENTS
