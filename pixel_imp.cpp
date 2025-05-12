#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

// Grayscale Conversion
Mat manualGrayscale(const Mat& image) {
    Mat gray(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            uchar grayVal = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            gray.at<uchar>(i, j) = grayVal;
        }
    return gray;
}

// Rotate 90 Clockwise
Mat manualRotate90Clockwise(const Mat& img) {
    Mat rotated(img.cols, img.rows, img.type());
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            rotated.at<uchar>(j, img.rows - 1 - i) = img.at<uchar>(i, j);
    return rotated;
}

// Flip Vertically 
Mat manualFlipVertical(const Mat& img) {
    Mat flipped(img.size(), img.type());
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            flipped.at<uchar>(img.rows - 1 - i, j) = img.at<uchar>(i, j);
    return flipped;
}

// HSV Conversion (Manual - Only V channel)
Mat manualHSV(const Mat& gray) {
    Mat hsv(gray.rows, gray.cols, CV_8UC3);
    for (int i = 0; i < gray.rows; i++)
        for (int j = 0; j < gray.cols; j++) {
            uchar v = gray.at<uchar>(i, j);
            hsv.at<Vec3b>(i, j) = Vec3b(v,v,v); 
        }
    return hsv;
}

// Brightness Adjustment 
Mat manualBrightness(const Mat& img, int brightness) {
    Mat output = img.clone();
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            int value = img.at<uchar>(i, j) + brightness;
            output.at<uchar>(i, j) = saturate_cast<uchar>(value);
        }
    return output;
}

// Image Clipping 
Mat manualClip(const cv::Mat& img) {
    int x = 10;
    int y = 10;
    int width = img.cols - 20;  
    int height = img.rows - 20; 

    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Image is too small to clip 10 pixels from each boundary!" << std::endl;
        return img.clone(); 
    }

    Mat clipped(height, width, img.type());

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            if (img.channels() == 1) {
                clipped.at<uchar>(i, j) = img.at<uchar>(y + i, x + j);
            }
            else if (img.channels() == 3) {
                clipped.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(y + i, x + j);
            }
        }

    return clipped;
}

// Histogram Equalization
Mat manualEqualizeHist(const Mat& img) {
    int hist[256] = {0};
    int total = img.rows * img.cols;
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            hist[img.at<uchar>(i, j)]++;

    float cdf[256] = {0};
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++)
        cdf[i] = cdf[i - 1] + hist[i];

    for (int i = 0; i < 256; i++)
        cdf[i] = (cdf[i] - cdf[0]) / (total - 1) * 255;

    Mat result = img.clone();
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            result.at<uchar>(i, j) = static_cast<uchar>(cdf[img.at<uchar>(i, j)]);
    return result;
}

// Wiener Filtering 
Mat manualWienerFilter(const Mat& img, int kSize = 3) {
    Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);
    Mat result = img.clone();

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

    return result;
}

// Gaussian Filtering 
Mat manualGaussianFilter(const Mat& img, int kSize = 5, double sigma = 1.0) {
    int k = kSize / 2;
    Mat kernel(kSize, kSize, CV_64F);
    double sum = 0;

    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double val = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.at<double>(i + k, j + k) = val;
            sum += val;
        }
    }

    kernel /= sum;

    Mat result = img.clone();
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

    return result;
}

int main() {
    Mat image = imread("sample1.jpg");
    if (image.empty()) {
        cerr << "Error loading image\n";
        return -1;
    }

    double execTime[9] = {0};
    double t1, t2;

    t1 = getTickCount();
    Mat gray = manualGrayscale(image);
    t2 = getTickCount();
    execTime[0] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("1 - Grayscale", gray);
    waitKey(0);

    t1 = getTickCount();
    Mat rotated = manualRotate90Clockwise(gray);
    t2 = getTickCount(); 
    execTime[1] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("2 - Rotated", rotated);
    waitKey(0);

    t1 = getTickCount();
    Mat flipped = manualFlipVertical(rotated);
    t2 = getTickCount(); 
    execTime[2] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("3 - Flipped", flipped);
    waitKey(0);

    t1 = getTickCount();
    Mat hsv = manualHSV(gray);
    t2 = getTickCount();
    execTime[3] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("4 - HSV (only Value)", hsv);
    waitKey(0);

    t1 = getTickCount();
    Mat bright = manualBrightness(hsv,50);
    t2 = getTickCount(); 
    execTime[4] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("5 - Brightness +50", bright);
    waitKey(0);

    t1 = getTickCount();
    Mat clipped = manualClip(bright);
    t2 = getTickCount(); 
    execTime[5] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("6 - Clipped", clipped);
    waitKey(0);

    t1 = getTickCount();
    Mat equalized = manualEqualizeHist(clipped);
    t2 = getTickCount(); 
    execTime[6] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("7 - Histogram Equalized", equalized);
    waitKey(0);

    t1 = getTickCount();
    Mat wiener = manualWienerFilter(equalized);
    t2 = getTickCount();
    execTime[7] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("8 - Wiener Filter", wiener);
    waitKey(0);

    t1 = getTickCount();
    Mat gaussian = manualGaussianFilter(wiener);
    t2 = getTickCount(); 
    execTime[8] = (t2 - t1) * 1000 / getTickFrequency();
    imshow("9 - Gaussian Filter", gaussian);
    waitKey(0);

    destroyAllWindows();


        // Printing execution times
    std::cout << "Execution Time (ms):" << std::endl;
    std::cout << "1) Grayscale Conversion: " << execTime[0] << " ms" << std::endl;
    std::cout << "2) Rotate: " << execTime[1] << " ms" << std::endl;
    std::cout << "3) Flip: " << execTime[2] << " ms" << std::endl;
    std::cout << "4) HSV Conversion: " << execTime[3] << " ms" << std::endl;
    std::cout << "5) Brightness Adjustment: " << execTime[4] << " ms" << std::endl;
    std::cout << "6) Image Clipping: " << execTime[5] << " ms" << std::endl;
    std::cout << "7) Histogram Equalization: " << execTime[6] << " ms" << std::endl;
    std::cout << "8) Wiener Filtering: " << execTime[7] << " ms" << std::endl;
    std::cout << "9) Gaussian Filtering: " << execTime[8] << " ms" << std::endl;

            // At the end of main()
        std::ofstream outFile("pixel_implementation.csv");
        outFile << "Operation,Execution Time (ms)" << std::endl;
        outFile << "Grayscale Conversion," << execTime[0] << std::endl;
        outFile << "Rotate," << execTime[1] << std::endl;
        outFile << "Flip," << execTime[2] << std::endl;
        outFile << "HSV Conversion," << execTime[3] << std::endl;
        outFile << "Brightness Adjustment," << execTime[4] << std::endl;
        outFile << "Image Clipping," << execTime[5] << std::endl;
        outFile << "Histogram Equalization," << execTime[6] << std::endl;
        outFile << "Wiener Filtering," << execTime[7] << std::endl;
        outFile << "Gaussian Filtering," << execTime[8] << std::endl;
        outFile.close();

    return 0;
}

