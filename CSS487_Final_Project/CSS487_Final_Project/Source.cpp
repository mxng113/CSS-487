#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>

using namespace std;
using namespace cv;
int main()
{
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