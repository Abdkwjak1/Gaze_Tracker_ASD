import numpy as np
import os
import cv2
import time 
import warnings
from skimage import exposure
#import skvideo.io as skv
#from skimage.color import rgb2gray
#from skimage.transform import resize
from .unprojection import reproject
from .eyefitter import SingleEyeFitter
from .utils import save_json, load_json, convert_vec2angle31
from .visualisation import  VideoManager ,Draw_frame,Montage


class gaze_inferer(object):
    def __init__(self, flen, ori_video_shape, sensor_size, infer_gaze_flag=True):
        """
        Initialize necessary parameters 
        
        Args:
            model: Deep learning model that perform image segmentation. Pre-trained model is provided at https://github.com/pydsgz/DeepVOG/model/DeepVOG_model.py, simply by loading load_DeepVOG() with "DeepVOG_weights.h5" in the same directory. If you use your own model, it should take input of grayscale image (m, 240, 320, 3) with value float [0,1] and output (m, 240, 320, 3) with value float [0,1] where (m, 240, 320, 1) is the pupil map.
            
            flen (float): Focal length of camera in mm. You can look it up at the product menu of your camera
            
            ori_video_shape (tuple or list or np.ndarray): Original video shape from your camera, (height, width) in pixel. If you cropped the video before, use the "original" shape but not the cropped shape
            
            sensor_size (tuple or list or np.ndarray): Sensor size of your camera, (height, width) in mm. For 1/3 inch CMOS sensor, it should be (3.6, 4.8). Further reference can be found in https://en.wikipedia.org/wiki/Image_sensor_format and you can look up in your camera product menu
        

        """
        # Assertion of shape
        try:
            assert ((isinstance(flen, int) or isinstance(flen, float)))
            assert (isinstance(ori_video_shape, tuple) or isinstance(ori_video_shape, list) or isinstance(
                ori_video_shape, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size,
                                                                                                  np.ndarray))
        except AssertionError:
            print("At least one of your arguments does not have correct type")
            raise TypeError

        # Parameters dealing with camera and video shape
        self.pupil_cascade = cv2.CascadeClassifier("cascade93.xml")
        self.flen = flen
        self.ori_video_shape, self.sensor_size = np.array(ori_video_shape).squeeze(), np.array(sensor_size).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape) / np.linalg.norm(self.sensor_size)
        self.confidence_fitting_threshold = 90
        self.eyefitter = SingleEyeFitter(focal_length=self.flen * self.mm2px_scaling,
                                         pupil_radius=2 * self.mm2px_scaling,
                                         initial_eye_z=50 * self.mm2px_scaling)
        self.infer_gaze_flag = infer_gaze_flag
        
        
    def process(self, video_src="live", mode='default', output_record_path="",
                output_video_path="", calibration_samples=100,vis=False):
        """

        Parameters
        ----------
        video_src : str
            Path of the video from which you want to (1) fit the eyeball model or (2) infer the gaze.
        mode : str
            There are two modes: "Fit" or "Infer". "Fit" will fit an eyeball model from the video source.
            "Infer" will infer the gaze from the video source.
        batch_size : int
            Batch size. Recommended >= 32.
        output_record_path : str
            Path of the csv file of your gaze estimation result. Only matter if the mode == "Infer".
        output_video_path : str
            Path of the output visualization video. If mode == "Fit", it draws segmented pupil ellipse.
            If mode == "Infer", it draws segmented pupil ellipse and gaze vector. if output_video_path == "",
            no visualization will be produced.

        Returns
        -------
        None

        """
        
        # Correct eyefitter's parameters in accordance with the image resizing
        self.eyefitter.focal_length = self.flen * self.mm2px_scaling * 0.5
        self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * 0.5
        
        if video_src=="live":
            print("Starting Live Eye_tracking Session.....")
            # Eye Camera :
            self.vreader =  cv2.VideoCapture(2)
            # World Camera :
            self.world = cv2.VideoCapture(0)
            # Get video information (path strings, frame reader, video's shapes...etc)
            #video_name_root, ext, vreader, vid_shapes, shape_correct, image_scaling_factor = self._get_video_info(video_src)
            #(vid_m, vid_w, vid_h, vid_channels) = vid_shapes
            
            self.vid_manager = VideoManager(vreader=self.vreader, output_record_path=output_record_path,
                                            output_video_path=output_video_path,vis=vis)

            # Check if the eyeball model is imported
            if mode == "Infer":
                self._check_eyeball_model_exists()

            # Setting Parameters for Processing the Video
            self.cal_flag=False
            self.cal_idx=0
            self.blink_flag=False
            self.blink_count=0
            self.Cas_x=0
            self.Cas_y=0
            self.fps=0
            newtime=time.time()
            oldtime = counter = self.fps=0
            # Start the process
            while(self.vreader.isOpened()):
                grabbed,frame=self.vreader.read()
                if grabbed:
                    
                    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                    #back=np.zeros((vid_w,vid_h),np.uint8)
                    #x,y,frame_y = self._pupil_segmenter(frame,self.pupil_cascade,back,mode="Abd")
                    #frame = cv2.resize(frame, (320,240))
                    #frame_y = cv2.resize(frame_y, (320, 240))
                
                    #Create fps count :
                    newtime=time.time()
                    counter+=1
                    if ((newtime-oldtime) >= 1):
                        oldtime=time.time()
                        self.fps=counter
                        counter=0

                    if mode == "default":
                        self._default_batch(frame)
                    if mode == "Fit" and not self.cal_flag:
                        self._fitting_batch(frame,calibration_samples)
                    if mode == "Infer":
                        self._infer_batch(frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cv2.destroyAllWindows()
            
        else:
            # Get video information (path strings, frame reader, video's shapes...etc)
            video_name_root, ext, vreader, vid_shapes, shape_correct, image_scaling_factor = self._get_video_info(video_src)
            (vid_m, vid_w, vid_h, vid_channels) = vid_shapes

            self.vid_manager = VideoManager(vreader=vreader, output_record_path=output_record_path,
                                            output_video_path=output_video_path,vis=vis)

            # Check if the eyeball model is imported
            if mode == "Infer":
                self._check_eyeball_model_exists()

            # Setting Parameters for Processing the Video
            self.cal_flag=False
            self.cal_idx=0
            self.blink_flag=False
            self.Cas_x=0
            self.Cas_y=0
            self.fps=0
            newtime=time.time()
            oldtime = counter = self.fps=0
            # Start the process
            while(vreader.isOpened()):
                grabbed,frame=vreader.read()
                if grabbed:
                    #back=np.zeros((vid_w,vid_h),np.uint8)
                    #x,y,frame_y = self._pupil_segmenter(frame,self.pupil_cascade,back,mode="Abd")
                    #frame = cv2.resize(frame, (320,240))
                    #frame_y = cv2.resize(frame_y, (320, 240))
                
                    #Create fps count :
                    newtime=time.time()
                    counter+=1
                    if ((newtime-oldtime) >= 1):
                        oldtime=time.time()
                        self.fps=counter
                        counter=0

                    if mode == "default":
                        self._default_batch(frame)
                    if mode == "Fit" and not self.cal_flag:
                        self._fitting_batch(frame,calibration_samples)
                    if mode == "Infer":
                        self._infer_batch(frame)
                    
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            cv2.destroyAllWindows()
                            

    
    
    def _default_batch(self, frame):
        
        
        _,Wframe=self.world.read()
        
        pupil,pupil_bw = self._pupil_segmenter(frame,self.pupil_cascade,mode="Abd")
        
        if not self.blink_flag:
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pupil_bw,self.Cas_x,self.Cas_y)
            ( centre, w, h, radian, ellipse_confidence) = ellipse_info
            
            if centre is not None:
                # Draw ellipse and pupil centre on input video if visualization is enabled
                if self.vid_manager.vis:
                    Draw_frame(frame,ellipse_info)
                    
        
        Montage(eye=frame,world=Wframe,pupil=pupil,pupil_bw=pupil_bw,blink=self.Blinking(),fps=self.fps)

    def _fitting_batch(self, frame,cal_samps):

        pupil,pupil_bw = self._pupil_segmenter(frame,self.pupil_cascade,mode="Abd")
        
        if not self.blink_flag:
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pupil_bw,self.Cas_x, self.Cas_y)
            ( centre, w, h, radian, ellipse_confidence) = ellipse_info

            # Fit each observation to eyeball model
            if centre is not None:
                
                if (ellipse_confidence >self.confidence_fitting_threshold):
                    self.eyefitter.add_to_fitting()
                    self.cal_idx +=1 
                # Draw ellipse and pupil centre on input video if visualization is enabled
                if self.vid_manager.vis:
                    Draw_frame(frame,ellipse_info)
        
        cv2.putText(frame,"FPS: {}".format(self.fps),(25,460), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,10),1,cv2.LINE_AA)            
        frame=cv2.resize(frame, (320,240))       
        cv2.imshow("OUTPUT",frame)
                    
        if self.cal_idx == cal_samps:
            print ("After Collecting Data >>> Starting 3D Eye Model Calibration")
             # Fit eyeball models. Parameters are stored as internal attributes of Eyefitter instance.
            _ = self.eyefitter.fit_projected_eye_centre(ransac=True, max_iters=100, min_distance=10*cal_samps)
            _, _ = self.eyefitter.estimate_eye_sphere(self.Cas_x, self.Cas_y)
            if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
                raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")
            else:
                print("Done !!!")
                self.cal_flag=True
                                
            
    def _infer_batch(self, frame):
        
        positions, gaze_angles, inference_confidence = None, None, None
        _,Wframe=self.world.read()
        pupil,pupil_bw = self._pupil_segmenter(frame,self.pupil_cascade,mode="Abd")
        
        if not self.blink_flag:
            
            _, _, _, _, ellipse_info = self.eyefitter.unproject_single_observation(pupil_bw,self.Cas_x, self.Cas_y)
            ( centre, w, h, radian, ellipse_confidence) = ellipse_info
            

            # If ellipse fitting is successful, i.e. an ellipse is located, AND gaze inference is ENABLED
            if (centre is not None) and self.infer_gaze_flag:
                p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
                p1, _ = p_list[0], n_list[0]
                px, py, pz = p1[0, 0], p1[1, 0], p1[2, 0]
                vec_with_length =(p1 - self.eyefitter.eye_centre).reshape(3,1)
                vec_with_length = vec_with_length / np.linalg.norm(vec_with_length)
                x, y = convert_vec2angle31(vec_with_length)
                positions = (px, py, pz, centre[0], centre[1])  # Pupil 3D positions and 2D projected positions
                gaze_angles = (x, y)  # horizontal and vertical gaze angles
                inference_confidence = (ellipse_confidence, consistence)
               
                
                if self.vid_manager.output_record_flag:
                    self.vid_manager.write_results(frame_id=frame, pupil2D_x=centre[0], pupil2D_y=centre[1], gaze_x=x,
                                                   gaze_y=y, confidence=ellipse_confidence, consistence=consistence)

                if self.vid_manager.vis:
                    # # Code below is for drawing video
                    projected_eye_centre = reproject(self.eyefitter.eye_centre,
                                                     self.eyefitter.focal_length)  # shape (2,1)
                    
                    Draw_frame(frame,ellipse_info,proj_eye_cent=projected_eye_centre,
                               gaze_flag=self.infer_gaze_flag,gaze_vec=vec_with_length)
                   
                    xx=int(np.round(self.mm2px_scaling*np.cos(x)))+320
                    yy=int(np.round(self.mm2px_scaling*np.cos(y)))+240
                    cv2.circle(Wframe,(xx,yy),10,(255,0,0),-1)
                    #print(self.eyefitter.eye_centre, projected_eye_centre)

            # If ellipse fitting is successful, i.e. an ellipse is located, AND gaze inference is DISABLED
            elif (centre is not None) and (not self.infer_gaze_flag):
                if self.vid_manager.output_record_flag:
                    self.vid_manager.write_results(frame_id=frame, pupil2D_x=centre[0], pupil2D_y=centre[1], gaze_x=np.nan,
                                                   gaze_y=np.nan, confidence=ellipse_confidence, consistence=np.nan)
                if self.vid_manager.vis:
                     Draw_frame(frame,ellipse_info)

                #self.vid_manager.write_results(frame_id=frame, pupil2D_x=np.nan, pupil2D_y=np.nan, gaze_x=np.nan,
                  #                             gaze_y=np.nan, confidence=np.nan, consistence=np.nan)

        
        Montage(eye=frame,world=Wframe,pupil=pupil,pupil_bw=pupil_bw,blink=self.Blinking(),
                fps=self.fps,gaze_angles=gaze_angles,gaze_flag=self.infer_gaze_flag)

        return positions, gaze_angles, inference_confidence
    
    def _pupil_segmenter(self,frame,cascade,mode="default"):
        frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frw, frh = frame.shape[1], frame.shape[0]
        w, h = frw, frh
        pupils = cascade.detectMultiScale(frame,1.50,9)
        # Count detected pupil candidates
        if len(pupils) > 0:
            # Use largest pupil candidate (works most of the time)
            sizes = np.sqrt(pupils[:,2] * pupils[:,3])
            best_pupil = sizes.argmax()
            # Get ROI info for largest pupil
            self.Cas_x, self.Cas_y, w, h = pupils[best_pupil,:]
            self.blink_flag=False
        else:
            # Dummy ROI during blink
            self.Cas_x, self.Cas_x, w, h = 0, 0, 1, 1
            self.blink_flag = True
            
        if w < 1 or h < 1:
            self.Cas_x, self.Cas_y, w, h = 0, 0, 1, 1
        
        roi = frame[self.Cas_y:self.Cas_y+h, self.Cas_x:self.Cas_x+w]
        roi_1 = np.zeros_like(roi)
        
        if not self.blink_flag:
            pA, pB = np.percentile(roi, (5,95))
            if pB != pA:
                roi = exposure.rescale_intensity(roi, in_range=(pA, pB))
            
            iterr=10
            acc=0.8
            k= 2
        
            Z = np.float32(roi.reshape((-1,1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterr, acc)
            _,_,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            thresh=int((center[0]+center[1])/2)
            pupil_bw =cv2.inRange(roi,0,thresh)
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            pupil_bw = cv2.morphologyEx(pupil_bw, cv2.MORPH_CLOSE, kernel1)
            
            if pupil_bw.sum() == 0:
                self.blink_flag=True
                
                
        
            if mode=="default":
                back=np.zeros((240,320),np.uint8)
                back[self.Cas_y:self.Cas_y+h, self.Cas_x:self.Cas_x+w] = pupil_bw
                #back = cv2.resize(back, (320, 240), interpolation=cv2.INTER_AREA)
                cv2.imshow("segmanted",back)
                return back
            elif mode =="Abd":
                #cv2.imshow("segmented",pupil_bw)
                return roi,pupil_bw 
        else:
            return (roi_1,roi_1)
        
    def _get_video_info(self, video_src):
        video_name_with_ext = os.path.split(video_src)[1]
        video_name_root, ext = os.path.splitext(video_name_with_ext)
        vreader =  cv2.VideoCapture(video_src)
        m,w,h, channels = (vreader.get(cv2.CAP_PROP_FRAME_COUNT)),480,640,3
        image_scaling_factor = np.linalg.norm((480, 640)) / np.linalg.norm((h, w))
        shape_correct = self._inspectVideoShape(w, h)
        return video_name_root, ext, vreader, (m, w, h, channels), shape_correct, image_scaling_factor

    def _check_eyeball_model_exists(self):
        try:
            if self.infer_gaze_flag:
                assert isinstance(self.eyefitter.eye_centre, np.ndarray)
                assert self.eyefitter.eye_centre.shape == (3, 1)
                assert self.eyefitter.aver_eye_radius is not None
            else:
                pass
        except AssertionError as e:
            print(
                "3D eyeball mode is not found. Gaze inference cannot continue. Please fit/load an eyeball model first")
            raise e
    
    def save_eyeball_model(self, path):
        """
        Save eyeball model parameters in json format.
        
        Args:
            path (str): path of the eyeball model file.
        """

        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            print("3D eyeball model not found. You may need -v or -m argument to check whether the pupil segmentation works properly.")
            raise
        else:
            save_dict = {"eye_centre": self.eyefitter.eye_centre.tolist(),
                         "aver_eye_radius": self.eyefitter.aver_eye_radius}
            save_json(path, save_dict)

    def load_eyeball_model(self, path):
        """
        Load eyeball model parameters of json format from path.
        
        Args:
            path (str): path of the eyeball model file.
        """
        loaded_dict = load_json(path)
        if (self.eyefitter.eye_centre is not None) or (self.eyefitter.aver_eye_radius is not None):
            warnings.warn("3D eyeball exists and reloaded")

        self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]
    
    def Blinking(self):
        if self.blink_flag:
            self.blink_count =self.blink_count + 1
        else:
            self.blink_count =0
            
        if self.blink_count >= 3 :
            self.blink_count=0
            return True
        else:
            return False
    
    @staticmethod
    def _inspectVideoShape(w, h):
        if (w, h) == (240, 320):
            return True
        else:
            return False

    @staticmethod
    def _computeCroppedShape(ori_video_shape, crop_size):
        video = np.zeros(ori_video_shape)
        cropped = video[crop_size[0]:crop_size[1], crop_size[2], crop_size[3]]
        return cropped.shape

    

    


