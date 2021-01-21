//
// Created by vyomkesh jha on 27.12.2020.
//
#include "headers/fast_fourier.h"

int main(int argc, char** argv)
{
    fast_fourier *transformer = new fast_fourier();

    if (argc != 3)
    {
        cerr << "usage: FFT <size of array> <number of threads>" << endl;
        exit(1);
    }

    int n = atoi(argv[1]); //Assign size of array
    if (!(n > 0) || !((n & (n - 1)) == 0)) //Test if
    {
        cerr << "usage: <size of array> must be power of 2" << endl;
        exit(1);
    }

    int p = atoi(argv[2]); //Assign number of threads to use
    double log2_of_n = log10(n) / log10(2); //Set for use later
    double *xReal = new double[n];
    double *xImg = new double[n];
    double *yReal = new double[n];
    double *yImg = new double[n];
    double *wReal = new double[(int) log2_of_n];
    double *wImg = new double[(int) log2_of_n];
    double *W_OracleR = new double[n / 2];
    double *W_OracleI = new double[n / 2];

    omp_set_num_threads(p); //Set how many proc's to user
    transformer->fill_array(n, xReal, xImg, yReal, yImg);
    timespec sTime;
    clock_gettime(CLOCK_MONOTONIC, &sTime); //Set start time
    transformer->perform_bit_reversal(n, xReal, xImg, log2_of_n);
    transformer->get_twiddle_factors(log2_of_n, n, wReal, wImg);
    transformer->calculate_fft_par(log2_of_n, xReal, xImg, wReal, wImg, n);
    timespec fTime;
    clock_gettime(CLOCK_MONOTONIC, &fTime); //Set end time
    double frac = 1.0 / (double) n; //Set fraction var
    double error = 0.0;
    for (int i = 0; i < n; i++)
    {   //Iterate through values and compile error with FFT ( FFT ( X(1:N) ) ) == N * X(1:N)
        //Thus we want close to zero as possible
        error = error + pow(yReal[i] - frac * xReal[i], 2) + pow(yImg[i] - frac * xImg[i], 2);
    }
    error = sqrt(frac * error); //Calculate final error
    cout << "Parallel FFT Time-elapsed: " << transformer->get_time_taken(sTime, fTime).tv_sec
         << "." << transformer->get_time_taken(sTime, fTime).tv_nsec << " seconds; ";// << endl;
    cout << "Error (Mine): " << error << endl;

    return 0;
}
