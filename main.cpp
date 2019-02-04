#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// fastest is cv::filter2D
void mySharpen(const Mat& myImage, Mat& result);

void myHorizontalEdgeDetector(const Mat& myImage, Mat& result);

void myVerticalEdgeDetector(const Mat& myImage, Mat& result);

void myImageBlender(const Mat& myImage1, const Mat& myImage2, Mat& result,
	double alpha = 0.5);

void myManualImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha);

int main()
{
	Mat src1, src2, dst0, dst1;

	src1 = imread("..\\..\\Images\\Desert.jpg", IMREAD_COLOR);
	src2 = imread("..\\..\\Images\\Koala.jpg", IMREAD_COLOR);

	namedWindow("Input 1", WINDOW_AUTOSIZE);
	namedWindow("Input 2", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);
//	namedWindow("Output H", WINDOW_AUTOSIZE);
//	namedWindow("Output V", WINDOW_AUTOSIZE);

	imshow("Input 1", src1);
	imshow("Input 2", src2);

//	myImageBlender(src1, src2, dst0);
	myManualImageBlender(src1, src2, dst0, 0.5);
//	myHorizontalEdgeDetector(src1, dst0);
//	myVerticalEdgeDetector(src1, dst1);
//	mySharpen(src, dst0);

	imshow("Output", dst0);
//	imshow("Output H", dst0);
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
			*output++ = (1.0 - alpha)*current1[i] + alpha*current2[i];
	}
}

void myImageBlender(const Mat& mySrc1, const Mat& mySrc2, Mat& result,
	double alpha)
{
	addWeighted(mySrc1, alpha, mySrc2, 1.0 - alpha, 0.0, result);
}