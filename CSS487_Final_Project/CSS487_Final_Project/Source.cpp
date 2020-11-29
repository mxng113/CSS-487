#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>

using namespace std;
using namespace cv;
void overlayFilter(Mat& sourceImage, Mat& overlay, int row, int col, double scale, Rect area);
void eyeDetection(CascadeClassifier classifier, const Mat& frame, const Mat& resizedFrame, const Rect& area, const double& scale, Point& center);

int main()
{
	Mat overlay = imread("headband_overlay.jpg");
	Mat sunglasses = imread("sunglasses_overlay.jpg");
	double scale = 2.0;
	CascadeClassifier faceCascade, leftEyeCascade, rightEyeCascade;

	// Load classifiers for face detection
	try {
		faceCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");		// for cj
		leftEyeCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_mcs_lefteye.xml");
		rightEyeCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_righteye_2splits.xml");
		//faceCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");	// for max
		//leftEyeCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_lefteye_2splits.xml");
		//rightEyeCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_righteye_2splits.xml");
	}
	// display error message if file did not load
	catch (Exception e) {
		if (faceCascade.empty()) {
			cout << "Couldn't load classifiers" << endl;
			exit(1);
		}
	}

	VideoCapture cap(0);

	for (;;)
	{
		Mat frame;
		Mat greyscale;
		Mat resizedFrame;

		cap >> frame;
		cvtColor(frame, greyscale, COLOR_BGR2GRAY);
		resize(greyscale, resizedFrame, Size(greyscale.size().width / scale, greyscale.size().height / scale));

		// To improve the contrast and brightneses of image
		equalizeHist(resizedFrame, resizedFrame);

		vector<Rect> faces;
		faceCascade.detectMultiScale(resizedFrame, faces, 1.1, 3, 0, Size(30, 30));

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

			Point leftEye, rightEye;

			eyeDetection(leftEyeCascade, frame, resizedFrame, area, scale, leftEye);
			eyeDetection(rightEyeCascade, frame, resizedFrame, area, scale, rightEye);

			overlayFilter(frame, sunglasses, (leftEye.y + rightEye.y) / 2, (leftEye.x + rightEye.x) / 2, scale, area);

			//// Detect left eye
			//Mat faceImg = resizedFrame(area);	// Store the face
			//vector<Rect> leftEyes;	// Initialize left eyes vectors
			//Point center;			// Center of the face
			//int radius;

			//leftEyeCascade.detectMultiScale(faceImg, leftEyes, 1.1, 3, 0, Size(30, 30));

			////cout << "Eyes detected: " << leftEyes.size() << endl;;

			//for (Rect eye : leftEyes) {

			//	center.x = cvRound((area.x + eye.x + eye.width * 0.5) * scale);
			//	center.y = cvRound((area.y + eye.y + eye.height * 0.5) * scale);
			//	radius = cvRound((eye.width + eye.height) * 0.25 * scale);
			//	circle(frame, center, radius, drawColor, 3);
			//}


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
				if (newPixel[0] < 254 || newPixel[1] < 254 || newPixel[2] < 254)
				{
					sourceImage.at<Vec3b>(abs(scaleY - area.height) + i, scaleX + j) = newPixel;
				}
			}
		}
	}

}

void eyeDetection(CascadeClassifier classifier, const Mat& frame, const Mat& resizedFrame, const Rect& area, const double& scale, Point& center) {

	Mat faceImg = resizedFrame(area);	// Store the face
	vector<Rect> eyesVector;	// Initialize left eyes vectors
	//Point center;			// Center of the face
	int radius;

	classifier.detectMultiScale(faceImg, eyesVector, 1.1, 3, 0, Size(30, 30));

	//cout << "Eyes detected: " << leftEyes.size() << endl;;

	// Draw blue circle around the eye
	for (Rect eye : eyesVector) {

		center.x = cvRound((area.x + eye.x + eye.width * 0.5) * scale);
		center.y = cvRound((area.y + eye.y + eye.height * 0.5) * scale);
		radius = cvRound((eye.width + eye.height) * 0.25 * scale);
		circle(frame, center, radius, Scalar(0, 0, 255), 3);
	}
}