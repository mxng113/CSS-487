#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>

using namespace std;
using namespace cv;
void overlayFilter(Mat& sourceImage, Mat& overlay, int row, int col, double scale, Rect area);
int main()
{
	Mat overlay = imread("headband_overlay.jpg");
	double scale = 2.0;
	CascadeClassifier faceCascade;
	faceCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");
	VideoCapture cap(0);

	for (;;)
	{
		Mat frame;
		Mat greyscale;
		cap >> frame;
		cvtColor(frame, greyscale, COLOR_BGR2GRAY);
		resize(greyscale, greyscale, Size(greyscale.size().width / scale, greyscale.size().height / scale));

		vector<Rect> faces;
		faceCascade.detectMultiScale(greyscale, faces, 1.1, 3, 0, Size(30, 30));
		for (Rect area : faces)
		{
			
			cout << area.x << '\t';
			cout << area.y;
			cout << endl;
			Mat resized;
			//double scaleX = area.width / overlay.cols;
			//double scaleY = area.height / overlay.rows;
			resize(overlay, resized, Size(area.width * scale, area.height * 1.5), 3);

			overlayFilter(frame, resized, area.y, area.x, scale, area);
			//imshow("overlayResize", resized);
			Scalar drawColor = (Scalar(0, 255, 0));
			rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
				Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)), drawColor);

		}
		imshow("webcam frame", frame);
		if (waitKey(30) >= 0)
		{
			break;
		}
	}

	return 0;

}

void overlayFilter(Mat& sourceImage, Mat& overlay, int row, int col, double scale, Rect area)
{
	int scaleY = cvRound(row * scale);
	int scaleX = cvRound(col * scale);
	if (row > overlay.rows)
	{
		for (int i = 0; i < overlay.rows; i++)
		{
			for (int j = 0; j < overlay.cols; j++)
			{
				Vec3b newPixel = overlay.at<Vec3b>(i, j);
				if (newPixel[0] < 255 || newPixel[1] < 255 || newPixel[2] < 255)
				{
					sourceImage.at<Vec3b>(abs(scaleY - area.height) + i, scaleX + j) = newPixel;
				}
			}
		}
	}

}