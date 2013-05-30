package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";     
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
   
    private CascadeClassifier[]      mJavaDetectors = new CascadeClassifier[2]; 
    private DetectionBasedTracker[]  mNativeDetectors = new DetectionBasedTracker[2];

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    private Random mRandom = new Random();
    

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");
                    loadDetectors(R.raw.lbpcascade_frontalface,"lbpcascade_frontalface.xml", 0); 
                    loadDetectors(R.raw.haarcascade_eye,"haarcascade_eye.xml", 1);

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
    
    
    private void loadDetectors(int rawid , String fileName  , int num ){
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(rawid);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, fileName);
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetectors[num] = new CascadeClassifier(cascadeFile.getAbsolutePath());
            if (mJavaDetectors[num].empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetectors[num] = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());

            mNativeDetectors[num] = new DetectionBasedTracker(cascadeFile.getAbsolutePath(), 0);

            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
        
    	
    }
    

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_5, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetectors[0].setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetectors[0] != null)
                mJavaDetectors[0].detectMultiScale(mGray, faces, 1.1, 2, 2,          
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetectors[0]!= null)
                mNativeDetectors[0].detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++){
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            // find eyes in founded faces
            Mat faceROI = mGray.submat(facesArray[i]);
            MatOfRect eyes = new MatOfRect();
            // launch detector
            if (mDetectorType == JAVA_DETECTOR) {
                if (mJavaDetectors[1] != null)
                    mJavaDetectors[1].detectMultiScale(faceROI, eyes, 1.1, 2, 2, new Size(30, 30), new Size());
            }
            else if (mDetectorType == NATIVE_DETECTOR) {
                if (mNativeDetectors[1]!= null)
                    mNativeDetectors[1].detect(faceROI, eyes);
            }
            else {
                Log.e(TAG, "Detection method is not selected!");
            }
            //draw eyes
            Rect[] eyesArray  = eyes.toArray();
            for( int j = 0; j < eyesArray.length; j++ ){
            	Point center = new Point(facesArray[i].x+eyesArray[j].x+eyesArray[j].width/2, 
            							facesArray[i].y+eyesArray[j].y+eyesArray[j].height/2);//

	            int radius =  (eyesArray[j].width + eyesArray[j].height)/4;
	            Core.circle(mRgba, center, radius, new Scalar(255, 255, 255, 255), -1  );
	            
	            // add crazy look to face in front
	            int crazyradius = mRandom.nextInt(radius/3)+radius/3;
	            int crazyrange = (int) ((radius-crazyradius)/1.4f);
	            int crazyXMargin = mRandom.nextInt(crazyrange*2)-crazyrange;
	            int crazyYMargin = mRandom.nextInt(crazyrange*2)-crazyrange;
	            Point crazyCenter = new Point(center.x+crazyXMargin,center.y+crazyYMargin);
	            
	            Core.circle(mRgba, crazyCenter, (int) (crazyradius), new Scalar(0, 0, 0, 255), -1  );
	            
            }
            
        } 

        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            mDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[mDetectorType]);
            setDetectorType(mDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                
                mNativeDetectors[0].start();
                mNativeDetectors[1].start();
                mNativeDetectors[1].setMinFaceSize(30);
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetectors[1].stop();
                mNativeDetectors[1].stop();
            }
        }
    }
}
