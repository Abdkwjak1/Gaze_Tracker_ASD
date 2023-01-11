from skimage.draw import ellipse_perimeter, line, circle_perimeter, line_aa
import skvideo.io as skv
import numpy as np
import cv2

def draw_line(output_frame, frame_shape, o, l, color=[255, 0, 0]):
    """

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    o : list or tuple or numpy.darray
        Origin of the line, with shape (2,) denoting (x, y).
    l : list or tuple or numpy.darray
        Vector with length. Body of the line. Shape = (2, ), denoting (x, y)
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame with the ellipse drawn.
    """
    R, G, B = color
    rr, cc = line(int(np.round(o[0])), int(np.round(o[1])), int(np.round(o[0] + l[0])), int(np.round(o[1] + l[1])))
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_ellipse(output_frame, frame_shape, ellipse_info, color=[255, 255, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    ellipse_info : list or tuple
        Information of ellipse parameters. (rr, cc, centre, w, h, radian, ellipse_confidence).
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the ellipse drawn.
    """

    R, G, B = color
    (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_circle(output_frame, frame_shape, centre, radius, color=[255, 0, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    centre : list or tuple or numpy.darray
        x,y coordinate of the circle centre
    radius : int or float
        Radius of the circle to draw.
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the circle drawn.
    """

    R, G, B = color
    rr_p1, cc_p1 = circle_perimeter(int(np.round(centre[0])), int(np.round(centre[1])), radius)
    rr_p1[rr_p1 > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc_p1[cc_p1 > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr_p1[rr_p1 < 0] = 0
    cc_p1[cc_p1 < 0] = 0
    output_frame[cc_p1, rr_p1, 0] = R
    output_frame[cc_p1, rr_p1, 1] = G
    output_frame[cc_p1, rr_p1, 2] = B
    return output_frame


def Montage(eye=None,world=None,pupil=None,pupil_bw=None,blink=False,fps=1,gaze_angles=None,gaze_flag=False):
    if blink:
        cv2.putText(eye,"BLINKING",(15,215), cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,50),1,cv2.LINE_AA)
    if gaze_flag:
        cv2.putText(eye,"Gaze angles:{}".format(gaze_angles),(20,30), cv2.FONT_HERSHEY_SIMPLEX,0.25,(100,50,50),1,cv2.LINE_AA)
    cv2.putText(eye,"FPS: {}".format(fps),(15,230), cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,10),1,cv2.LINE_AA)
    
    #cv2.imshow("OUTPUT",eye)
    #cv2.imshow("WORLD",world)
    
    pupil_bw=_touint8(pupil_bw)
    
    A=np.hstack((pupil,pupil_bw))
    A = cv2.applyColorMap(A, cv2.COLORMAP_BONE)
    A=cv2.resize(A,(320,240),interpolation=cv2.INTER_NEAREST)
    
    B=np.vstack((eye,A))
    Final=np.hstack((world,B))
    
    cv2.imshow('Pupilometry',Final)
    
def Draw_frame(frame,ellipse,proj_eye_cent=None,gaze_flag=False,gaze_vec=None):
    
    
    ( centre, w, h, radian, ellipse_confidence) = ellipse
    # Draw pupil ellipse
    cv2.ellipse(frame, (centre,(w,h),radian), (0,0,255), 2)
    # Draw small circle at the ellipse centre
    cv2.circle(frame,centre,5,(255,0,0),2)
    # Draw Ellipse Confidence
    cv2.putText(frame,"Ellipse Confidence:{}".format(ellipse_confidence),(20,20), cv2.FONT_HERSHEY_SIMPLEX,0.25,(100,50,50),1,cv2.LINE_AA)
    
    if gaze_flag:
        
        ellipse_centre_np = np.array(centre)
        # The lines below are for translation from camera coordinate system (centred at image centre)
        # to numpy's indexing frame. You substract the vector by the half of the video's 2D shape.
        # Col = x-axis, Row = y-axis
        vid_frame_shape_2d=frame.shape[0],frame.shape[1]
        #projected_eye_centre += np.array(vid_frame_shape_2d[::-1]).reshape(-1, 1) / 2
        #n1 += np.array(vid_frame_shape_2d[::-1]).reshape(-1, 1) / 2
        #print(projected_eye_centre)
        # Draw from eyeball centre to ellipse centre (just connecting two points)
        cv2.line(frame,(int(proj_eye_cent[0]),int(proj_eye_cent[1])),centre,(255,255,0),2)
        
        # Draw gaze vector originated from ellipse centre
        #cv2.line(frame,centre,(n1[1]*700,n1[0]*700),(255,100,100),2)
        frame = draw_line(output_frame=frame, frame_shape=vid_frame_shape_2d, o=ellipse_centre_np,
                          l=gaze_vec*50 , color=[255, 0, 0])

def _touint8(x):
    '''
    Rescale and cast arbitrary number x to uint8
    '''

    x = np.float32(x)

    x_min, x_max = x.min(), x.max()

    y = ((x - x_min) / (x_max - x_min)) * 255.0

    return np.uint8(y)


    
    
        
class VideoManager:
    def __init__(self, vreader, output_record_path="", output_video_path="", vis=False):
        # Parameters
        self.vreader = vreader
        self.output_video_flag = True if output_video_path else False
        self.output_record_flag = True if output_record_path else False
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')        
        self.vwriter = cv2.VideoWriter(output_video_path,fourcc, 20.0, (320,240)) if self.output_video_flag else None
        self.results_recorder = open(output_record_path, "w") if self.output_record_flag else None
        self.vis=vis
        # Initialization actions
        self._initialize_results_recorder()


    def write_results(self, frame_id, pupil2D_x, pupil2D_y, gaze_x, gaze_y, confidence, consistence):
        self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame_id, pupil2D_x, pupil2D_y,
                                                                gaze_x, gaze_y,
                                                                confidence, consistence))

    def _initialize_results_recorder(self):
        if self.output_record_flag:
            self.results_recorder.write("frame,pupil2D_x,pupil2D_y,gaze_x,gaze_y,confidence,consistence\n")

    def __del__(self):
        #self.vreader.close()
        self.vreader.release()
        
        if self.vwriter:
            self.vwriter.close()
        if self.results_recorder:
            self.results_recorder.close()
