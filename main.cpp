#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

/*
TODO
- add lookup tables where needed
- pass things by reference to avoid copying
*/

enum KType { Gaussian, LaplacianOfGaussian };

enum EdgeStrength { NonEdge, Weak, Strong };

void myNegative(const Mat& myImage, Mat& result);
// fastest is cv::filter2D
void mySharpen(const Mat& myImage, Mat& result);

void myHorizontalSobel(const Mat& myImage, Mat& result);

void myVerticalSobel(const Mat& myImage, Mat& result);

void myXYSobel(const Mat& myImage, Mat& result);

void myImageBlender(const Mat& myImage1, const Mat& myImage2, Mat& result,
	double alpha = 0.5);

void myManualImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha);

void myBoxFilter(const Mat& myImage, Mat& result);

void myContrastAndBrightness(const Mat& myImage, Mat& myResult,
	double alpha, double beta);

void myGammaCorrection(const Mat& myImage, Mat& result, double gamma);
// only works for grayscale
void myDft(const Mat& myImage, Mat& result);

Mat_<double> getKernel(Size ksize, int sigX, int sigY, bool highPass,
	KType kType);

void myManualGaussian(const Mat& myImage, Mat& result, Size kSize, int sigX,
	int sigY, bool highPass);

void myMedianFilter(const Mat& myImage, Mat& result, Size kSize);

void myManualLaplacian(const Mat& myImage, Mat& result, Size ksize);

void myLaplacianOfGaussian(const Mat& myImage, Mat& result, Size kSize,
	int sigX, int sigY, bool highPass);

void myCanny(const Mat& myImage, Mat result[], Size kSize, int sigX, int sigY,
	int weakThreshold, int strongThreshold);

int myQuickSelect(std::vector<uchar>& list, int left, int right, int k);

int partition(std::vector<uchar>& list, int left, int right);

int main()
{
	Mat src1, src2, src3, src4, src1Smaller;
	Mat dst[6];

	if (!DEBUG_MODE)
	{
		src1 = imread("..\\..\\Images\\Desert.jpg",
			IMREAD_GRAYSCALE);
		src2 = imread("..\\..\\Images\\Koala.jpg",
			IMREAD_GRAYSCALE);
		src3 = imread("..\\..\\Images\\Noise_salt_and_pepper.png",
			IMREAD_GRAYSCALE);
	}
	else
	{
		src1 = imread("..\\Images\\Desert.jpg",
			IMREAD_GRAYSCALE);
		src2 = imread("..\\Images\\Koala.jpg",
			IMREAD_GRAYSCALE);
		src3 = imread("..\\Images\\Noise_salt_and_pepper.png",
			IMREAD_GRAYSCALE);
	}

	namedWindow("All", WINDOW_AUTOSIZE);

	resize(src1, src1Smaller, Size(), 0.40, 0.40);
	myCanny(src1Smaller, dst, Size(3, 3), 1, 1, 30, 80);

	Mat h[2];
	Mat out;
	hconcat(dst, 3, h[0]);
	hconcat(dst+3, 3, h[1]);
	vconcat(h, 2, out);
	imshow("All", out);

	waitKey();
	return 0;
}

void myNegative(const Mat& myImage, Mat& result)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 0; j < myImage.rows; ++j)
	{
		const uchar* current = myImage.ptr<uchar>(j);
		uchar* output = result.ptr<uchar>(j);

		for (int i = 0; i < nChannels*myImage.cols; ++i)
			*output++ = 255 - current[i];
	}
}

void mySharpen(const Mat& myImage, Mat& result)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * current[i]
				- current[i - nChannels] - current[i + nChannels]
				- previous[i] - next[i]);
		}
	}
	result.row(0).setTo(Scalar(0));
	result.row(result.rows-1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols-1).setTo(Scalar(0));
}

void myHorizontalSobel(const Mat& myImage, Mat& result)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(next[i-nChannels] + 2*next[i]
				+ next[i+nChannels] - (previous[i-nChannels] + 2*previous[i]
				+ previous[i+nChannels]));
		}
	}
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void myVerticalSobel(const Mat& myImage, Mat& result)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(previous[i+nChannels]
				+ 2*current[i+nChannels] + next[i+nChannels]
				- (previous[i-nChannels] + 2*current[i-nChannels]
				+ next[i+nChannels]));
		}
	}
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void myXYSobel(const Mat& myImage, Mat& result)
{
	result.create(myImage.size(), myImage.type());

	Mat temp1, temp2;

	myHorizontalSobel(myImage, temp1);
	myVerticalSobel(myImage, temp2);

	addWeighted(temp1, 0.5, temp2, 0.5, 0, result);
}

void myManualImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha)
{
	if (mySrc1.rows != mySrc2.rows || mySrc2.cols != mySrc2.cols)
	{
		cout << "Images not same dimensions" << endl;
		return;
	}

	if (mySrc1.channels() != mySrc2.channels())
	{
		cout << "Images not same number of channels" << endl;
		return;
	}

	CV_Assert(mySrc1.depth() == CV_8U); // accept only uchar images
	CV_Assert(mySrc2.depth() == CV_8U); // accept only uchar images

	const int nChannels = mySrc1.channels();
	result.create(mySrc1.size(), mySrc1.type());

	for (int j = 0; j < mySrc1.rows; ++j)
	{
		const uchar* current1 = mySrc1.ptr<uchar>(j);
		const uchar* current2 = mySrc2.ptr<uchar>(j);
		uchar* output = result.ptr<uchar>(j);

		for (int i = 0; i < nChannels*mySrc1.cols; ++i)
			*output++ = saturate_cast<uchar>((1.0 - alpha)*current1[i]
				+ alpha*current2[i]);
	}
}

void myImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha)
{
	addWeighted(mySrc1, alpha, mySrc2, 1.0 - alpha, 0.0, result);
}

void myBoxFilter(const Mat& myImage, Mat& result)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = result.ptr<uchar>(j);

		double oneNinth = 1 / 9.0;
		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(oneNinth*(previous[i-nChannels]
				+ previous[i] + previous[i+nChannels] + current[i-nChannels]
				+ current[i] + current[i+nChannels] + next[i-nChannels]
				+ next[i] + next[i+nChannels]));
		}
	}
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
}

void myContrastAndBrightness(const Mat& myImage, Mat& result,
	double alpha, double beta)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 0; j < myImage.rows; ++j)
	{
		const uchar* current = myImage.ptr<uchar>(j);
		uchar* output = result.ptr<uchar>(j);

		for (int i = 0; i < nChannels*myImage.cols; ++i)
			*output++ = saturate_cast<uchar>(alpha*current[i] + beta);
	}
}

void myGammaCorrection(const Mat& myImage, Mat& result, double gamma)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 0; j < myImage.rows; ++j)
	{
		const uchar* current = myImage.ptr<uchar>(j);
		uchar* output = result.ptr<uchar>(j);

		for (int i = 0; i < nChannels*myImage.cols; ++i)
			*output++ = saturate_cast<uchar>(
				pow(current[i]/255.0, gamma) * 255.0);
	}
}

void myDft(const Mat& myImage, Mat& result)
{
	// pad input image with zeros for faster DFT
	Mat padded;
	int m = getOptimalDFTSize(myImage.rows);
	int n = getOptimalDFTSize(myImage.cols);
	copyMakeBorder(myImage, padded, 0, m - myImage.rows, 0, n - myImage.cols,
		BORDER_CONSTANT, Scalar::all(0));

	// convert Mat elements to float to hold larger frequency domain values
	// and add second channel to hold imaginary values
	Mat channels[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(channels, 2, complexI);

	dft(complexI, complexI);

	// compute magnitude
	split(complexI, channels);
	magnitude(channels[0], channels[1], channels[0]);
	result = channels[0];

	// switch to log scale
	result += Scalar::all(1);
	log(result, result);

	// crop spectrum due to padded zeros at beginning for faster DFT
	result = result(Rect(0, 0, result.cols & -2, result.rows & -2));

	// rearrange quadrants so origin is at center
	int cx = result.cols / 2;
	int cy = result.rows / 2;

	Mat q0(result, Rect(0, 0, cx, cy)); // Top-Left
	Mat q1(result, Rect(cx, 0, cx, cy)); // Top-Right
	Mat q2(result, Rect(0, cy, cx, cy)); // Bottom-Left
	Mat q3(result, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;
	q0.copyTo(tmp); // swap quadrants
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp); // swap quadrants
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(result, result, 0, 1, NORM_MINMAX);
}

Mat_<double> getKernel(Size kSize, int sigX, int sigY, bool highPass,
	KType kType)
{
	// compute x and y means
	const uchar xMean = (kSize.width - 1) / 2;
	const uchar yMean = (kSize.height - 1) / 2;

	// compute x and y spreads
	const double xSpread = 1 / (2 * (pow(sigX, 2)));
	const double ySpread = 1 / (2 * (pow(sigY, 2)));

	// x part of kernel
	Mat_<double> gaussX(1, kSize.width);
	int x;
	for (int i = 0; i < kSize.width; ++i)
	{
		x = i - xMean;
		gaussX.col(i) = exp(-pow(x, 2) * xSpread);
	}

	// combine with y to get kernel
	Mat_<double> kernel(kSize.height, kSize.width);
	int y;
	for (int i = 0; i < kSize.height; ++i)
	{
		y = i - yMean;
		kernel.row(i) = gaussX * exp(-pow(y, 2) * ySpread);
	}

	if (kType == KType::LaplacianOfGaussian)
	{
		for (int row = 0; row < kSize.height; ++row)
		{
			for (int col = 0; col < kSize.width; ++col)
			{
				y = row - yMean;
				x = col - xMean;
				kernel.at<double>(row, col) *=
					(((pow(x, 2) - pow(sigX, 2)) / pow(sigY, 4)) +
					((pow(y, 2) - pow(sigY, 2)) / pow(sigY, 4)));
			}
		}
	}

	// normalize
	double denominator = 0;
	for (auto k : kernel)
		denominator += k;
	kernel = kernel / denominator;

	// high pass gaussian
	if (highPass)
	{
		for (int row = 0; row < kSize.height; ++row)
			for (int col = 0; col < kSize.width; ++col)
				if (row == (kSize.height - 1) / 2
					&& col == (kSize.width - 1) / 2)
					kernel.at<double>(row, col) =
					1 - kernel.at<double>(row, col);
				else
					kernel.at<double>(row, col) =
					-1 * kernel.at<double>(row, col);
	}

	return kernel;
}

void myManualGaussian(const Mat& myImage, Mat& result, Size kSize, int sigX,
	int sigY, bool highPass)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	// compute x and y means
	const uchar xMean = (kSize.width - 1) / 2;
	const uchar yMean = (kSize.height - 1) / 2;

	Mat_<double> kernel = getKernel(kSize, sigX, sigY, highPass,
		KType::Gaussian);

	// take channels into account (so both gray and color images work)
	const uchar nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = yMean; j < myImage.rows - yMean; ++j)
	{
		const uchar* imageCurrent;
		uchar* output = result.ptr<uchar>(j);

		for (int i = xMean*nChannels;
			i < (myImage.cols - xMean)*nChannels;++i)
		{
			for (int kJ = -yMean; kJ < yMean + 1; ++kJ)
			{
				imageCurrent = myImage.ptr<uchar>(j + kJ);
				for (int kI = -xMean; kI < xMean + 1; ++kI)
					*output += saturate_cast<uchar>(
						imageCurrent[i + kI * nChannels]
						* kernel[kJ + yMean][kI + xMean]);
			}
			++output;
		}
	}
}

void myMedianFilter(const Mat& myImage, Mat& result, Size kSize)
{
	// compute x and y midpoints
	const uchar xMid = (kSize.width - 1) / 2;
	const uchar yMid = (kSize.height - 1) / 2;

	// take channels into account (so both gray and color images work)
	const uchar nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	uchar chCopy;
	std::vector<uchar> buffer;

	for (int j = yMid; j < myImage.rows - yMid; ++j)
	{
		const uchar* imageCurrent;
		uchar* output = result.ptr<uchar>(j);

		for (int i = xMid*nChannels;
			i < (myImage.cols - xMid)*nChannels; ++i)
		{
			for (int kJ = -yMid; kJ < yMid + 1; ++kJ)
			{
				imageCurrent = myImage.ptr<uchar>(j + kJ);
				for (int kI = -xMid; kI < xMid + 1; ++kI)
				{
					chCopy = imageCurrent[i + kI * nChannels];
					buffer.push_back(chCopy);
				}
			}
			*output++ = myQuickSelect(buffer, 0, kSize.width*kSize.height - 1,
				kSize.width*kSize.height / 2);
			buffer.clear();
		}
	}
}

void myManualLaplacian(const Mat& myImage, Mat& result, Size ksize)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	const int nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);

		uchar* output = result.ptr<uchar>(j);

		for (int i = nChannels; i < nChannels*(myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(previous[i - nChannels]
				+ previous[i] + previous[i + nChannels] + current[i - nChannels]
				- 8*current[i] + current[i + nChannels] + next[i - nChannels]
				+ next[i] + next[i + nChannels]);
		}

		result.row(0).setTo(Scalar(0));
		result.row(result.rows - 1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols - 1).setTo(Scalar(0));
	}
}

void myLaplacianOfGaussian(const Mat& myImage, Mat& result, Size kSize,
	int sigX, int sigY, bool highPass)
{

	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

	// compute x and y means
	const uchar xMean = (kSize.width - 1) / 2;
	const uchar yMean = (kSize.height - 1) / 2;

	Mat_<double> kernel = getKernel(kSize, sigX, sigY, highPass,
		KType::LaplacianOfGaussian);

	cout << "kernel: \n" << kernel << endl;

	// take channels into account (so both gray and color images work)
	const uchar nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = yMean; j < myImage.rows - yMean; ++j)
	{
		const uchar* imageCurrent;
		uchar* output = result.ptr<uchar>(j);

		for (int i = xMean * nChannels;
			i < (myImage.cols - xMean)*nChannels; ++i)
		{
			for (int kJ = -yMean; kJ < yMean + 1; ++kJ)
			{
				imageCurrent = myImage.ptr<uchar>(j + kJ);
				for (int kI = -xMean; kI < xMean + 1; ++kI)
					*output += saturate_cast<uchar>(
						imageCurrent[i + kI * nChannels]
						* kernel[kJ + yMean][kI + xMean]);
			}
			++output;
		}
	}
}

/*
Broke canny up into separate stages to observe each stage's effect. Could
be made much more efficient by combining the stages.
*/
void myCanny(const Mat& myImage, Mat result[], Size kSize, int sigX, int sigY,
	int weakThreshold, int strongThreshold)
{
	result[0] = myImage;

	// Smoothing
	GaussianBlur(myImage, result[1], kSize, sigX, sigY);

	// Gradients
	Mat gX, gY;
	Sobel(result[1], gX, CV_16S, 1, 0);
	Sobel(result[1], gY, CV_16S, 0, 1);

	Mat magnitude(gX.size(), gX.type());
	Mat direction(gX.size(), gX.type());
	Mat edgeStrength(gX.size(), CV_8U);
	short x, y;
	short mag;
	short arctanVal;
	short rounded;
	for (int col = 0; col < gX.cols; ++col)
	{
		for (int row = 0; row < gX.rows; ++row)
		{
			x = gX.at<short>(row, col);
			y = gY.at<short>(row, col);
			mag = sqrt(pow(x, 2) + pow(y, 2));

			if (mag < weakThreshold)
				edgeStrength.at<uchar>(row, col) = EdgeStrength::NonEdge;
			else if (mag > strongThreshold)
				edgeStrength.at<uchar>(row, col) = EdgeStrength::Strong;
			else
				edgeStrength.at<uchar>(row, col) = EdgeStrength::Weak;

			magnitude.at<short>(row, col) = mag;
			arctanVal = cvFastArctan(y, x);
			rounded = ((short)(arctanVal + 22.5) / 45) * 45;
			if (rounded > 135) rounded -= 180;
			direction.at<short>(row, col) = rounded;
		}
	}

	convertScaleAbs(magnitude, result[2]);

	// Non-maximal suppression
	short curDir, curMag;
	bool zeroOut = false;
	for (int row = 0; row < magnitude.rows; ++row)
	{
		for (int col = 0; col < magnitude.cols; ++col)
		{
			curMag = magnitude.at<short>(row, col);
			curDir = direction.at<short>(row, col);
			if (curDir == 0) // check left, right
			{
				if (col + 1 < magnitude.cols) // check right bound
					if (curMag < magnitude.at<short>(row, col + 1))
						zeroOut = true;
				if (col - 1 >= 0) // check left bound
					if (curMag < magnitude.at<short>(row, col - 1))
						zeroOut = true;
			}
			else if (curDir == 45) // check upright, downleft
			{
				if (row - 1 >= 0 && col + 1 < magnitude.cols) // check upright bound
					if (curMag < magnitude.at<short>(row - 1, col + 1))
						zeroOut = true;
				if (row + 1 < magnitude.rows && col - 1 >= 0) // check downleft bound
					if (curMag < magnitude.at<short>(row + 1, col - 1))
						zeroOut = true;
			}
			else if (curDir == 90) // check up, down
			{
				if (row - 1 >= 0) // check up bound
					if (curMag < magnitude.at<short>(row - 1, col))
						zeroOut = true;
				if (row + 1 < magnitude.rows) // check down bound
					if (curMag < magnitude.at<short>(row + 1, col))
						zeroOut = true;
			}
			else // check upleft, downright
			{
				if (row - 1 >= 0 && col - 1 >= 0) // check upleft bound
					if (curMag < magnitude.at<short>(row - 1, col - 1))
						zeroOut = true;
				if (row + 1 < magnitude.rows && col + 1 < magnitude.cols) // check downright bound
					if (curMag < magnitude.at<short>(row + 1, col + 1))
						zeroOut = true;
			}
			if (zeroOut)
			{
				magnitude.at<short>(row, col) = 0;
				zeroOut = false;
			}
		}
	}

	convertScaleAbs(magnitude, result[3]);

	// Double thresholding
	uchar edge;
	for (int row = 0; row < edgeStrength.rows; ++row)
	{
		for (int col = 0; col < edgeStrength.cols; ++col)
		{
			edge = edgeStrength.at<uchar>(row, col);
			if (edge == EdgeStrength::NonEdge)
				magnitude.at<short>(row, col) = 0;
			else if (edge == EdgeStrength::Strong)
				magnitude.at<short>(row, col) = 255;
			else
				magnitude.at<short>(row, col) = 50;
		}
	}

	convertScaleAbs(magnitude, result[4]);

	// Edge tracking by hysteresis
	bool keepEdge = false;
	for (int row = 0; row < edgeStrength.rows; ++row)
	{
		for (int col = 0; col < edgeStrength.cols; ++col)
		{
			edge = edgeStrength.at<uchar>(row, col);
			if (edge == EdgeStrength::Weak)
			{
				for (int i = -1; i < 2; ++i)
				{
					if (row + i >= 0 && row + i < edgeStrength.rows)
						for (int j = -1; j < 2; ++j)
						{
							if (i == 0 && j == 0) continue;
							if (col + i >= 0 && col + i < edgeStrength.cols)
								if (edgeStrength.at<uchar>(row + i, col + i)
									== EdgeStrength::Strong)
								{
									keepEdge = true;
									break;
								}
						}
					if (keepEdge)
						break;
				}
				if (keepEdge)
					keepEdge = false;
				else
					magnitude.at<short>(row, col) = 0;
			}
		}
	}

	convertScaleAbs(magnitude, result[5]);
}

int myQuickSelect(std::vector<uchar>& list, int left, int right, int k)
{
	int p = partition(list, left, right);
	if (k == p)
		return list[k];
	else if (k < p)
		return myQuickSelect(list, left, p - 1, k);
	else
		return myQuickSelect(list, p + 1, right, k);
}

int partition(std::vector<uchar>& list, int left, int right)
{
	int pivot = list[right], i = left, x;

	for (x = left; x < right; ++x)
	{
		if (list[x] < pivot)
		{
			swap(list[i], list[x]);
			++i;
		}
	}

	swap(list[i], list[right]);
	return i;
}