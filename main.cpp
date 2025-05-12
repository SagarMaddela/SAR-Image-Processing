#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat wienerFilter(const cv::Mat& src, int kernelSize = 5) {
    cv::Mat mean, meanSquare, variance;
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    cv::blur(srcFloat, mean, cv::Size(kernelSize, kernelSize));
    cv::blur(srcFloat.mul(srcFloat), meanSquare, cv::Size(kernelSize, kernelSize));
    variance = meanSquare - mean.mul(mean);
    
    double noiseVariance = cv::mean(variance)[0];

    cv::Mat result = mean + (variance - noiseVariance) / (variance + 1e-3).mul(srcFloat - mean);
    result.convertTo(result, CV_8U);
    return result;
}

int main() {
    cv::Mat image = cv::imread("sample1.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load the image!" << std::endl;
        return -1;
    }

    double t1 = 0, t2 = 0;
    double execTime[9] = {0};

    // 1) Grayscale Conversion
    t1 = cv::getTickCount();
    cv::Mat grayscaleImage;
    cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
    t2 = cv::getTickCount();
    execTime[0] = (t2 - t1) * 1000 / cv::getTickFrequency();
    cv::imwrite("grayscale.jpg", grayscaleImage);
    cv::imshow("Grayscale Image", grayscaleImage);
    cv::waitKey(0);  // Wait for a key press before moving to the next step
    cv::destroyAllWindows();

    // 2) Rotate
    t1 = cv::getTickCount();
    cv::Mat rotatedImage;
    cv::rotate(grayscaleImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
    t2 = cv::getTickCount();
    execTime[1] = (t2 - t1) * 1000 / cv::getTickFrequency();
    cv::imwrite("rotated.jpg", rotatedImage);

    cv::imshow("Rotated Image", rotatedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 3) Flip
    t1 = cv::getTickCount();
    cv::Mat flippedImage;
    cv::flip(rotatedImage, flippedImage, 0);
    t2 = cv::getTickCount();
    execTime[2] = (t2 - t1) * 1000 / cv::getTickFrequency();
    cv::imwrite("flipped.jpg", flippedImage);

    cv::imshow("Flipped Image", flippedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 4) HSV Conversion
    t1 = cv::getTickCount();
    cv::Mat gray3Channel, hsvImage;
    cv::cvtColor(grayscaleImage, gray3Channel, cv::COLOR_GRAY2BGR);
    cv::cvtColor(gray3Channel, hsvImage, cv::COLOR_BGR2HSV);
    t2 = cv::getTickCount();
    execTime[3] = (t2 - t1) * 1000 / cv::getTickFrequency();
    cv::imwrite("hsv.jpg", hsvImage);

    cv::imshow("HSV Image", hsvImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 5) Brightness Adjustment
    t1 = cv::getTickCount();
    cv::Mat brightImage;
    int brightness = 50;
    grayscaleImage.convertTo(brightImage, -1, 1, brightness);
    t2 = cv::getTickCount();
    execTime[4] = (t2 - t1) * 1000 / cv::getTickFrequency();
    // Save and display the brightened image
    cv::imwrite("brightness_adjusted.jpg", brightImage);
    cv::imshow("Brightness Adjusted Image", brightImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 6) Image Clipping
    // Print the dimensions of the image
    int imgWidth = brightImage.cols;
    int imgHeight = brightImage.rows;
    std::cout << "Image Width: " << imgWidth << std::endl;
    std::cout << "Image Height: " << imgHeight << std::endl;

    t1 = cv::getTickCount();
    int x = 10, y = 10, width = image.cols - 10, height = image.rows - 10;
    if (x + width > grayscaleImage.cols || y + height > grayscaleImage.rows) {
        std::cerr << "Error: ROI is out of bounds!" << std::endl;
        return -1;
    }
    cv::Mat clippedImage = brightImage(cv::Rect(x, y, width, height));
    t2 = cv::getTickCount();
    execTime[5] = (t2 - t1) * 1000 / cv::getTickFrequency();
    cv::imwrite("clipped.jpg", clippedImage);
    cv::imshow("Clipped Image", clippedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 7) Histogram Equalization
    t1 = cv::getTickCount();
    cv::Mat equalizedImage;
    cv::equalizeHist(clippedImage, equalizedImage);
    t2 = cv::getTickCount();
    execTime[6] = (t2 - t1) * 1000 / cv::getTickFrequency();
    // Save and display the equalized image
    cv::imwrite("equalized.jpg", equalizedImage);
    cv::imshow("Equalized Image", equalizedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 8) Wiener Filtering
    t1 = cv::getTickCount();
    cv::Mat wienerFiltered = wienerFilter(equalizedImage, 5);
    t2 = cv::getTickCount();
    execTime[7] = (t2 - t1) * 1000 / cv::getTickFrequency();
        // Save and display result
    cv::imwrite("wiener_filtered.jpg", wienerFiltered);
    cv::imshow("Wiener Filtered", wienerFiltered);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 9) Gaussian Filtering
    t1 = cv::getTickCount();
    cv::Mat gaussianFiltered;
    cv::GaussianBlur(wienerFiltered, gaussianFiltered, cv::Size(5, 5), 1.5, 1.5);
    t2 = cv::getTickCount();
    execTime[8] = (t2 - t1) * 1000 / cv::getTickFrequency();
        // Save and display the Gaussian filtered image
    cv::imwrite("gaussian_filtered.jpg", gaussianFiltered);
    cv::imshow("Gaussian Filtered", gaussianFiltered);
    cv::waitKey(0);
    cv::destroyAllWindows();


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

    return 0;
}

