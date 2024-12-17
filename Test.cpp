#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <random>

using namespace cv;
using namespace std;
 

void showImage(const Mat& image, const string& title) {
    Mat display;
    image.convertTo(display, CV_8U, 255.0);
    imshow(title, display);
    waitKey(0);
}

Mat sharpenImage(const Mat& src) {
    Mat result = src.clone();

    float kernel[3][3] = { { 0, -0.0625,  0 },
                           { -0.0625, 1.25, -0.0625 },
                           { 0, -0.0625,  0 } };

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            Vec3f newPixel = Vec3f(0, 0, 0);
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    Vec3f neighborPixel = src.at<Vec3f>(y + i, x + j);
                    newPixel += neighborPixel * kernel[i + 1][j + 1];
                }
            }
            result.at<Vec3f>(y, x) = newPixel;
        }
    }
    return result;
}

Mat adjustBrightnessContrast(const Mat& src, double alpha, double beta) {
    Mat result = src.clone();
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f pixel = src.at<Vec3f>(y, x);
            for (int c = 0; c < 3; c++) {
                pixel[c] = saturate_cast<float>(alpha * pixel[c] + beta);
            }
            result.at<Vec3f>(y, x) = pixel;
        }
    }
    return result;
}

Mat adjustSaturation(const Mat& src, double saturationFactor) {
    Mat result = src.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f pixel = src.at<Vec3f>(y, x);
            float maxVal = max({ pixel[0], pixel[1], pixel[2] });
            for (int c = 0; c < 3; c++) {
                pixel[c] = saturate_cast<float>(pixel[c] + (maxVal - pixel[c]) * saturationFactor);
            }
            result.at<Vec3f>(y, x) = pixel;
        }
    }
    return result;
}

int main() {
    string imagePath = "C:/Git/CPE 462 Project/tiger.jpg";
    //string imagePath = "C:\Git\CPE 462 Project\18609_1.png";
    Mat original = imread(imagePath, IMREAD_COLOR);
    if (original.empty()) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    original.convertTo(original, CV_32F, 1.0 / 255.0);
    showImage(original, "Original Image");

    double saturationFactor = -0.5; // Increase saturation by 5%
    Mat saturated = adjustSaturation(original, saturationFactor);
    showImage(saturated, "Saturation Adjusted");

    double alpha = 1.05; // Contrast control
    double beta = 0.1;  // Brightness control
    Mat brightnessContrastAdjusted = adjustBrightnessContrast(saturated, alpha, beta);
    showImage(brightnessContrastAdjusted, "Brightness & Contrast Adjusted");

    Mat sharpened = sharpenImage(brightnessContrastAdjusted);
    showImage(sharpened, "Sharpened Image");

    Mat enhanced = sharpened.clone();
    showImage(enhanced, "Final Enhanced Image");

    return 0;
}