#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h>
#include <filesystem>
namespace fs = std::filesystem;

using namespace cv;
using namespace std;

double execTime[9] = {0};
double t1, t2;

Mat Openmp_Grayscale(const Mat& image) {
        Mat gray(image.rows, image.cols, CV_8UC1);
        double start  = omp_get_wtime();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < image.rows; i++){
                for (int j = 0; j < image.cols; j++) {
                        //  std::cout << omp_get_num_threads() << std::endl;
                        Vec3b pixel = image.at<Vec3b>(i, j);
                        uchar grayVal = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
                        //      std::cout << "Thread " << omp_get_thread_num() << " computed " << i << "," << j << " pixel" << std::endl;
                        gray.at<uchar>(i, j) = grayVal;
                }
        }

        double end = omp_get_wtime();
        std::cout << "Openmp Grayscale Conversion :" << (end - start) * 1000 << std::endl;
        execTime[0] = (end - start ) * 1000;
        return gray;
}

// Rotate 90 Clockwise
Mat Openmp_Rotate90Clockwise(const Mat& img) {
        Mat rotated(img.cols, img.rows, img.type());
        double start  = omp_get_wtime();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < img.rows; i++)
                for (int j = 0; j < img.cols; j++)
                        rotated.at<uchar>(j, img.rows - 1 - i) = img.at<uchar>(i, j);

        double end = omp_get_wtime();
        std::cout << "Openmp Rotation :" << (end - start) * 1000 << std::endl;
        execTime[1] = (end - start ) * 1000;
        return rotated;
}

// Flip Vertically
Mat Openmp_FlipVertical(const Mat& img) {
        Mat flipped(img.size(), img.type());
        double start  = omp_get_wtime();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < img.rows; i++)
                for (int j = 0; j < img.cols; j++)
                        flipped.at<uchar>(img.rows - 1 - i, j) = img.at<uchar>(i, j);

        double end = omp_get_wtime();
        std::cout << "Openmp Flipping:" << (end - start) * 1000 << std::endl;
        execTime[2] = (end - start ) * 1000;
        return flipped;
}

// HSV Conversion (Manual - Only V channel)
Mat Openmp_HSV(const Mat& gray) {
        Mat hsv(gray.rows, gray.cols, CV_8UC3);
        double start  = omp_get_wtime();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < gray.rows; i++)
                for (int j = 0; j < gray.cols; j++) {
                        uchar v = gray.at<uchar>(i, j);
                        hsv.at<Vec3b>(i, j) = Vec3b(v,v, v);
                }

        double end = omp_get_wtime();
        std::cout << "Openmp HSV Conversion:" << (end - start) * 1000 << std::endl;
        execTime[3] = (end - start ) * 1000;
        return hsv;
}

// Brightness Adjustment
Mat Openmp_Brightness(const Mat& img, int brightness) {
        Mat output = img.clone();
        double start  = omp_get_wtime();
#pragma omp parallel for collapse(2)
        for (int i = 0; i < img.rows; i++)
                for (int j = 0; j < img.cols; j++) {
                        int value = img.at<uchar>(i, j) + brightness;
                        output.at<uchar>(i, j) = saturate_cast<uchar>(value);
                }

        double end = omp_get_wtime();
        std::cout << "Openmp Brightness:" << (end - start) * 1000 << std::endl;
        execTime[4] = (end - start ) * 1000;
        return output;
}

// Image Clipping
Mat Openmp_Clip(const cv::Mat& img) {
        int x = 10;
        int y = 10;
        int width = img.cols - 20;
        int height = img.rows - 20;

        if (width <= 0 || height <= 0) {
                std::cerr << "Error: Image is too small to clip 10 pixels from each boundary!" << std::endl;
                return img.clone();
        }

        Mat clipped(height, width, img.type());
        double start  = omp_get_wtime();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++) {
                        if (img.channels() == 1) {
                                clipped.at<uchar>(i, j) = img.at<uchar>(y + i, x + j);
                        }
                        else if (img.channels() == 3) {
                                clipped.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(y + i, x + j);
                        }
                }

        double end = omp_get_wtime();
        std::cout << "Openmp Clipping:" << (end - start) * 1000 << std::endl;
        execTime[5] = (end - start ) * 1000;
        return clipped;
}
// Histogram Equalization
Mat Openmp_histogram_equalization(const Mat& img) {
    Mat eq_img = img.clone();

    vector<int> histogram(256, 0);
    vector<int> lut(256, 0);
double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < eq_img.rows; ++i) {
        for (int j = 0; j < eq_img.cols; ++j) {
            uchar pixel = eq_img.at<Vec3b>(i, j)[0];
            #pragma omp atomic
            histogram[pixel]++;
        }
    }

    int total = eq_img.rows * eq_img.cols;
    int cum_sum = 0;
    for (int i = 0; i < 256; ++i) {
        cum_sum += histogram[i];
        lut[i] = static_cast<int>((cum_sum * 255.0) / total);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < eq_img.rows; ++i) {
     for (int j = 0; j < eq_img.cols; ++j) {
            uchar pixel = eq_img.at<Vec3b>(i, j)[0];
            uchar new_val = lut[pixel];
            eq_img.at<Vec3b>(i, j) = Vec3b(new_val, new_val, new_val);
        }
    }

    double end = omp_get_wtime();
    std::cout << "Openmp Histogram Equalization:" << (end - start) * 1000 << std::endl;
    execTime[6] = (end - start ) * 1000;
    return eq_img;
}

// Wiener Filtering
Mat Openmp_WienerFilter(const Mat& img, int kSize = 3) {
    Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);
    Mat result = img.clone();

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = kSize / 2; i < img.rows - kSize / 2; i++) {
        for (int j = kSize / 2; j < img.cols - kSize / 2; j++) {
            Rect roi(j - kSize / 2, i - kSize / 2, kSize, kSize);
            Mat patch = imgFloat(roi);

            Scalar mean, stddev;
            meanStdDev(patch, mean, stddev);
            float localVar = stddev[0] * stddev[0];
            float noiseVar = 0.0001f;

            float val = imgFloat.at<float>(i,j);
            float newVal = mean[0] + (max(localVar - noiseVar, 0.0f) / (localVar + 1e-5)) * (val - mean[0]);

            result.at<uchar>(i, j) = saturate_cast<uchar>(newVal);
        }
    }

    double end = omp_get_wtime();
    std::cout << "Openmp Wiener Filtering :" << (end - start) * 1000 << std::endl;
    execTime[7] = (end - start ) * 1000;
    return result;
}

Mat Openmp_GaussianFilter(const Mat& img, int kSize = 5, double sigma = 1.0) {
    int k = kSize / 2;
    Mat kernel(kSize, kSize, CV_64F);
    double sum = 0.0;

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double val = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + k, j + k) = val;
            sum += val;
        }
    }

    kernel /= sum;

    Mat result = img.clone();

    #pragma omp parallel for collapse(2)
    for (int i = k; i < img.rows - k; i++) {
        for (int j = k; j < img.cols - k; j++) {
            double pixel = 0.0;
            for (int m = -k; m <= k; m++) {
                for (int n = -k; n <= k; n++) {
                    pixel += img.at<uchar>(i + m, j + n) * kernel.at<double>(m + k, n + k);
                }
            }
            result.at<uchar>(i, j) = static_cast<uchar>(pixel);
        }
    }

    double end = omp_get_wtime();
    std::cout << "Openmp Guassian Filtering:" << (end - start) * 1000 << std::endl;
    execTime[8] = (end - start ) * 1000;
    return result;
}
int main() {
    std::string folderPath = "./images";  // Folder containing your 20-25 images
    std::ofstream csvFile("image_size_vs_time.csv");
    csvFile << "Image Name,Width,Height,Total Pixels,Total Execution Time (ms)\n";

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        Mat image = imread(entry.path().string());
        if (image.empty()) {
            std::cerr << "Failed to load " << entry.path().filename() << "\n";
            continue;
        }

        double execTime[9] = {0};
        double t1, t2;
        t1 = getTickCount();

 Mat gray = Openmp_Grayscale(image);
        Mat rotated = Openmp_Rotate90Clockwise(gray);
        Mat flipped = Openmp_FlipVertical(rotated);
        Mat hsv = Openmp_HSV(gray);
        Mat bright = Openmp_Brightness(hsv,50);
        Mat clipped = Openmp_Clip(bright);
        Mat equalized = Openmp_histogram_equalization(clipped);
        Mat wiener = Openmp_WienerFilter(equalized);
        Mat gaussian = Openmp_GaussianFilter(wiener);
        t2 = getTickCount();
        double totalTime = (t2 - t1) * 1000 / getTickFrequency();

        int width = image.cols;
        int height = image.rows;
        int totalPixels = width * height;

        csvFile << entry.path().filename() << "," << width << "," << height << "," 
                << totalPixels << "," << totalTime << "\n";

        std::cout << "Processed: " << entry.path().filename() 
                  << " | Time: " << totalTime << " ms\n";
    }

    csvFile.close();
    return 0;
}

