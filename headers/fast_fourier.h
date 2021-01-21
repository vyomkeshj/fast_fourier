//
// Created by vyomkesh jha on 27.12.2020.
//

#ifndef FAST_FOURIER_FAST_FOURIER_H
#define FAST_FOURIER_FAST_FOURIER_H

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <complex>

using namespace std;

class fast_fourier {

public:
    void perform_bit_reversal(int n, double *xReal, double *xImg, double log2_of_n);
    void fill_array(int n, double *xReal, double *xImg, double *yReal, double *yImg);
    void calculate_fft_par(int log2_of_n, double *xReal, double *xImg, double *wReal, double *wImg, int N);
    timespec get_time_taken(timespec s, timespec f);
    void get_twiddle_factors(int n, int N, double *wReal, double *wImg);
};


#endif //FAST_FOURIER_FAST_FOURIER_H
