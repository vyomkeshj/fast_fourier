//
// Created by vyomkesh jha on 27.12.2020.
//

#include "../headers/fast_fourier.h"

void fast_fourier::perform_bit_reversal(int n, double *xReal, double *xImg, double log2_of_n) {
    int i;
    double temp;
#pragma omp parallel for default(none) private(i, temp) shared(n, xReal, xImg, log2_of_n)
    for (i = 1; i < n; i++) {
        int k, rev = 0;
        int inp = i;
        for (k = 0; k < log2_of_n; k++) {
            rev = (rev << 1) | (inp & 1);
            inp >>= 1;
        }

        if (rev <= i) continue; //Skip if already done
        temp = xReal[i]; //Store into temp values and swap
        xReal[i] = xReal[rev];
        xReal[rev] = temp;
        temp = xImg[i];
        xImg[i] = xImg[rev];
        xImg[rev] = temp;
    }
}

void fast_fourier::fill_array(int n, double *xReal, double *xImg, double *yReal, double *yImg) {
    int i;
    struct drand48_data buffer;
    srand48_r(time(NULL) ^ omp_get_thread_num(), &buffer); //Seed for each thread
    //#pragma omp parallel for default(none) private(i) shared(x,n)
    for (i = 0; i < n; i++) {
        drand48_r(&buffer, &xReal[i]); //Store random double into xReal
        yReal[i] = xReal[i]; //Copy
        drand48_r(&buffer, &xImg[i]); //Store random double into xImg
    }

}

void fast_fourier::calculate_fft_par(int log2_of_n, double *xReal, double *xImg, double *wReal, double *wImg, int N) {
    int n, d, i, k;
    double temp_w, temp_x;
    for (n = 1; n < log2_of_n + 1; n++) {   //Iterate through time slice
        d = pow(2, n);
#pragma omp parallel for default(none) private(k, i, temp_w, temp_x) shared(n, N, d, xReal, xImg, wReal, wImg, log2_of_n)
        for (k = 0; k < (d / 2); k++) {   //Iterate through even and odd elements
            for (i = k; i < N; i += d) { //Butterfly operation
                temp_w = wReal[n - 1] * xReal[i + (d / 2)]; //Multiply by twiddle factor
                temp_x = xReal[i];  //Store in temp's and restore with correct values
                xReal[i] = temp_w + temp_x;
                xReal[i + (d / 2)] = temp_x - temp_w;
                temp_w = wImg[n - 1] * xImg[i + (d / 2)]; //Multiply by twiddle factor
                temp_x = xImg[i]; //Store in temp's and restore with correct values
                xImg[i] = temp_w + temp_x;
                xImg[i + (d / 2)] = temp_x - temp_w;
            }
        }
    }

}

timespec fast_fourier::get_time_taken(timespec s, timespec f) {
    timespec totTime;
    if ((f.tv_nsec - s.tv_nsec) < 0) //so we do not return a negative incorrect value
    {
        totTime.tv_sec = f.tv_sec - s.tv_sec - 1;
        totTime.tv_nsec = 1000000000 + f.tv_nsec - s.tv_nsec;
    } else {
        totTime.tv_sec = f.tv_sec - s.tv_sec;
        totTime.tv_nsec = f.tv_nsec - s.tv_nsec;
    }
    return totTime;

}

void fast_fourier::get_twiddle_factors(int log2_of_n, int n, double wReal[], double wImg[]) {
    int i;
#pragma omp parallel for default(none) private(i) shared(n, wReal, wImg, log2_of_n)
    for (i = 0; i < n; i++) {
        wReal[i] = cos(((double) i * 2.0 * M_PI) / ((double) n));
        wImg[i] = sin(((double) i * 2.0 * M_PI) / ((double) n));
    }

}
