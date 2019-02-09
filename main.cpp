#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

/*
add lookup tables to everything
*/

// fastest is cv::filter2D
void mySharpen(const Mat& myImage, Mat& result);

void myHorizontalEdgeDetector(const Mat& myImage, Mat& result);

void myVerticalEdgeDetector(const Mat& myImage, Mat& result);

void myImageBlender(const Mat& myImage1, const Mat& myImage2, Mat& result,
	double alpha = 0.5);

void myManualImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha);

void myNormalizedBoxFilter(const Mat& myImage, Mat& result);

void myContrastAndBrightness(const Mat& myImage, Mat& myResult,
	double alpha, double beta);

void myGammaCorrection(const Mat& myImage, Mat& result, double gamma);

void myDft(const Mat& myImage, Mat& result);

void myManualGaussianFilter(const Mat& myImage, Mat& result, Size kSize,
	int sigX, int sigY);

int main()
{
	Mat src1, src2, dst0, dst1;

	src1 = imread("..\\..\\Images\\Desert.jpg",
		IMREAD_GRAYSCALE);
	src2 = imread("..\\..\\Images\\Koala.jpg",
		IMREAD_GRAYSCALE);

	namedWindow("Input 1", WINDOW_AUTOSIZE);
//	namedWindow("Input 2", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);
//	namedWindow("Output H", WINDOW_AUTOSIZE);
//	namedWindow("Output V", WINDOW_AUTOSIZE);

	imshow("Input 1", src1);
//	imshow("Input 2", src2);

//	myImageBlender(src1, src2, dst0);
//	myManualImageBlender(src1, src2, dst0, 0.40);
//	myHorizontalEdgeDetector(src1, dst0);
//	myVerticalEdgeDetector(src1, dst0);
//	mySharpen(src1, dst0);
//	myNormalizedBoxFilter(src1, dst0);
//	myContrastAndBrightness(src1, dst0, 0, 5);
//	myGammaCorrection(src1, dst0, 0.5);
//	myDft(src1, dst0);
	myManualGaussianFilter(src1, dst0, Size(5, 5), 1, 1);
	GaussianBlur(src1, dst1, Size(5, 5), 1, 1);
	Mat x = getGaussianKernel(5, 1);
	Mat y = getGaussianKernel(5, 1);
	Mat k = x * y.t();
	cout << "cv 2d kernal: \n" << k << endl;

	imshow("Output Mine", dst0);
	imshow("Output", dst1);
//	imshow("Output V", dst1);

	waitKey();
	return 0;
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

		result.row(0).setTo(Scalar(0));
		result.row(result.rows-1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols-1).setTo(Scalar(0));
	}

}

void myHorizontalEdgeDetector(const Mat& myImage, Mat& result)
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

		result.row(0).setTo(Scalar(0));
		result.row(result.rows - 1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols - 1).setTo(Scalar(0));
	}
}

void myVerticalEdgeDetector(const Mat& myImage, Mat& result)
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

		result.row(0).setTo(Scalar(0));
		result.row(result.rows - 1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols - 1).setTo(Scalar(0));
	}
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

void myNormalizedBoxFilter(const Mat& myImage, Mat& result)
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

		result.row(0).setTo(Scalar(0));
		result.row(result.rows - 1).setTo(Scalar(0));
		result.col(0).setTo(Scalar(0));
		result.col(result.cols - 1).setTo(Scalar(0));
	}
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

void myManualGaussianFilter(const Mat& myImage, Mat& result, Size kSize,
	int sigX, int sigY)
{
	CV_Assert(myImage.depth() == CV_8U); // accept only uchar images

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
		gaussX.col(i) = exp(-pow(x,2) * xSpread);
	}

	// combine with y to get kernel
	Mat_<double> kernel(kSize.height, kSize.width);
	int y;
	for (int i = 0; i < kSize.height; ++i)
	{
		y = i - yMean;
		kernel.row(i) = gaussX * exp(-pow(y,2) * ySpread);
	}

	// normalize
	double denominator = 0;
	for (auto k : kernel)
		denominator += k;
	kernel = kernel / denominator;
	cout << "Kernel after normalization: \n" << kernel << endl;

	// take channels into account (so both gray and color images work)
	const uchar nChannels = myImage.channels();
	result.create(myImage.size(), myImage.type());

	for (int j = yMean; j < myImage.rows - 1 - yMean; ++j)
	{
		const uchar* imageCurrent;
		uchar* output = result.ptr<uchar>(j);

		for (int i = xMean*nChannels;
			i < (myImage.cols - 1 - xMean)*nChannels; ++i)
		{
			for (int kJ = 0; kJ < kSize.height - 1; ++kJ)
			{
				for (int kI = 0; kI < kSize.width - 1; ++kI)
				{
					imageCurrent = myImage.ptr<uchar>(j + kJ);
					*output += saturate_cast<uchar>(
						imageCurrent[i + kI] * kernel[kJ][kI]);
				}
			}
			++output;
		}
	}
}