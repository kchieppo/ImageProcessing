#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void myDisplayImage(int argc, char** argv);
// fastest is multithreaded cv::LUT
void myScanImageAndReduceC(Mat& I, const uchar* const table);
// fastest is cv::filter2D
void mySharpen(const Mat& myImage, Mat& result);

int main(int argc, char** argv)
{
//	myDisplayImage(argc, argv);

	Mat src, dst0, dst1;

	if (argc >= 3 && !strcmp("G", argv[2]))
		src = imread(argv[1], IMREAD_GRAYSCALE);
	else
		src = imread(argv[1], IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Couldn't open image" << endl;
	}

	namedWindow("Input", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);

	imshow("Input", src);

	mySharpen(src, dst0);

	imshow("Output", dst0);

	waitKey();
	return 0;
}

void myDisplayImage(int argc, char** argv)
{
	Mat image;

	if (argc >= 3 && !strcmp("G", argv[2]))
		image = imread(argv[1], IMREAD_GRAYSCALE);
	else
		image = imread(argv[1], IMREAD_COLOR);

	if (image.empty())
	{
		cout << "Couldn't open image" << endl;
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // create window for display
	imshow("Display window", image);
}

void myScanImageAndReduceC(Mat& I, const uchar* const table)
{

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