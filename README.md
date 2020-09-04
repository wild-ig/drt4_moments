# ImageMoments
Accurate calculation of higher order image moments with O(N+M) multiplications for grayscale images. 

### Extending The Discrete Radon Transformation
The DRT algorithm reduces this problem from 2D moments of an NxM array, to 1D moments of several projection arrays. The original image can be projected vertically, horizontally, at 45 & 135 degrees & others (e.g. 30 degrees, 60 degrees), and then summed along those axes. The raw moments then become linear combinations of 1D moments of these arrays and there is no loss of information. This reduces the number of multiplications from O(N.M) to O(N+M). See the timings on an Intel Core i5 7th Gen. The DRT algorithm and times are shown with and without SSE2/AVX optimisation: 
![Timings](https://github.com/wild-ig/drt_moments/raw/master/OpenCV4vsDRT4.png)