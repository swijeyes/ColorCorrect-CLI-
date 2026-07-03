#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace cv;
using namespace std;

// --- Color Balance (per-channel percentile stretch) ---
// Stretches a single channel's histogram so that its darkest ~clipPercent%
// of pixels become 0 and its brightest ~clipPercent% become 255, with
// everything in between linearly remapped across the full range. Clipping
// the extremes first keeps a few outlier pixels (dust, scratches, a bright
// highlight) from compressing the stretch for the rest of the image.
void stretchChannel(Mat& channel, double clipPercent) {
    const int histSize = 256;
    float rangeArr[] = {0, 256};
    const float* ranges[] = {rangeArr};
    const int channelsToUse[] = {0};

    Mat hist;
    calcHist(&channel, 1, channelsToUse, Mat(), hist, 1, &histSize, ranges);

    double totalPixels = channel.total();
    double lowThresh = totalPixels * (clipPercent / 100.0);
    double highThresh = totalPixels * (clipPercent / 100.0);

    // Walk up from 0 until the darkest clipPercent% of pixels are behind us
    int low = 0;
    double sum = 0.0;
    for (int i = 0; i < histSize; i++) {
        sum += hist.at<float>(i);
        if (sum > lowThresh) {
            low = i;
            break;
        }
    }

    // Walk down from 255 until the brightest clipPercent% of pixels are behind us
    int high = histSize - 1;
    sum = 0.0;
    for (int i = histSize - 1; i >= 0; i--) {
        sum += hist.at<float>(i);
        if (sum > highThresh) {
            high = i;
            break;
        }
    }

    // Degenerate case (e.g. a flat, single-value channel) - nothing to stretch
    if (high <= low) {
        return;
    }

    // dst = saturate_cast<uchar>(src * alpha + beta). Because the destination
    // is 8-bit, saturate_cast clamps anything below `low` to 0 and anything
    // above `high` to 255 automatically - so this one call both clips the
    // outlier pixels and stretches the rest across the full 0-255 range.
    double alpha = 255.0 / (high - low);
    double beta = -low * alpha;
    channel.convertTo(channel, CV_8U, alpha, beta);
}

// Applies the percentile stretch to each of B, G, R independently. Because
// each channel is stretched to fill 0-255 on its own, a channel that has
// faded (lower contrast, narrower range) gets pulled back out just as far
// as the others - correcting the color cast and boosting contrast in the
// same step.
void colorBalanceCorrection(Mat& img, double clipPercent) {
    vector<Mat> channels;
    split(img, channels);

    for (int i = 0; i < 3; i++) {
        stretchChannel(channels[i], clipPercent);
    }

    merge(channels, img);
}

// --- Gray World Assumption ---
// Scales each channel so its mean intensity matches the overall gray mean.
void grayWorldCorrection(Mat& img) {
    vector<Mat> channels;
    split(img, channels);

    double meanB = mean(channels[0])[0];
    double meanG = mean(channels[1])[0];
    double meanR = mean(channels[2])[0];

    double meanGray = (meanB + meanG + meanR) / 3.0;

    double scaleB = meanGray / meanB;
    double scaleG = meanGray / meanG;
    double scaleR = meanGray / meanR;

    channels[0] *= scaleB;
    channels[1] *= scaleG;
    channels[2] *= scaleR;

    Mat merged;
    merge(channels, merged);
    merged.convertTo(img, CV_8UC3);
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " <image_path> [options]" << endl;
    cout << "Options:" << endl;
    cout << "  -g, --gray-world       Use the gray world method instead of the default color balance method" << endl;
    cout << "  -p, --percent <value>  Percent of pixels to clip on each end for color balance (default: 0.5)" << endl;
    cout << "  -h, --help             Show this help message" << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }

    string inputPath;
    bool useGrayWorld = false;
    double clipPercent = 0.5;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-g" || arg == "--gray-world") {
            useGrayWorld = true;
        } else if (arg == "-p" || arg == "--percent") {
            if (i + 1 >= argc) {
                cout << "Error: " << arg << " requires a value" << endl;
                return -1;
            }
            clipPercent = atof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (inputPath.empty()) {
            inputPath = arg;
        } else {
            cout << "Error: unrecognized argument '" << arg << "'" << endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    if (inputPath.empty()) {
        printUsage(argv[0]);
        return -1;
    }

    if (clipPercent < 0.0 || clipPercent >= 50.0) {
        cout << "Error: --percent must be between 0 and 50" << endl;
        return -1;
    }

    // Read the vintage image
    Mat img = imread(inputPath, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: Could not open or find the image " << inputPath << endl;
        return -1;
    }

    // --- Step 1: Color Cast Correction ---
    Mat colorCorrected = img.clone();
    if (useGrayWorld) {
        grayWorldCorrection(colorCorrected);
        cout << "Using gray world color correction." << endl;
    } else {
        colorBalanceCorrection(colorCorrected, clipPercent);
        cout << "Using color balance correction (clipping " << clipPercent << "% per channel end)." << endl;
    }

    // --- Step 2: Local Contrast Enhancement ---
    // Convert to LAB color space to adjust lightness without shifting colors again
    Mat labImg;
    cvtColor(colorCorrected, labImg, COLOR_BGR2Lab);

    vector<Mat> labChannels;
    split(labImg, labChannels);

    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Lightness channel
    // A clip limit of 2.0 keeps the contrast smooth and natural
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(labChannels[0], labChannels[0]);

    merge(labChannels, labImg);

    // Convert back to BGR for saving
    Mat finalImg;
    cvtColor(labImg, finalImg, COLOR_Lab2BGR);

    // --- Step 3: Save the Output ---
    // Create a new filename for the corrected image
    size_t lastSlash = inputPath.find_last_of("/\\");
    string filename = (lastSlash == string::npos) ? inputPath : inputPath.substr(lastSlash + 1);
    string outputPath = "restored_" + filename;

    imwrite(outputPath, finalImg);

    cout << "Restoration complete. Image saved as: " << outputPath << endl;

    return 0;
}
