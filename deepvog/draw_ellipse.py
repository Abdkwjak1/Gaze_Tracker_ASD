import numpy as np
import matplotlib as mpl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from .bwperim import bwperim
from .ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting
from skimage.draw import ellipse_perimeter
import cv2

def isolate_islands(prediction, threshold):
    bw = closing(prediction > threshold , square(3))
    labelled = label(bw)  
    regions_properties = regionprops(labelled)
    max_region_area = 0
    select_region = 0
    for region in regions_properties:
        if region.area > max_region_area:
            max_region_area = region.area
            select_region = region
    output = np.zeros(labelled.shape)
    if select_region == 0:
        return output
    else:
        output[labelled == select_region.label] = 1
        return output

# input: output from bwperim -- 2D image with perimeter of the ellipse = 1
def gen_ellipse_contour_perim(perim, color = "r"): 
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:,1], vertices[:,0]])
            center, w,h, radian = fitted.parameters()
            ell = mpl.patches.Ellipse(xy = [center[0],center[1]], width = w*2, height = h*2, angle = np.rad2deg(radian), fill = False, color = color)
            # Because of the np indexing of y-axis, orientation needs to be minus
            rr, cc = ellipse_perimeter(int(np.round(center[0])), int(np.round(center[1])), int(np.round(w)), int(np.round(h)), -radian)
            return (rr, cc, center, w, h, radian, ell)
        except:
            return None

def gen_ellipse_contour_perim_compact(perim): 
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:,1], vertices[:,0]])
            center, w,h, radian = fitted.parameters()
            # Because of the np indexing of y-axis, orientation needs to be minus
            return (center, w,h, radian)
        except:
            return None

def fit_ellipse(img, threshold = 0.5, color = "r", mask=None,method="default"):
    

    if method=="default":
        
        isolated_pred = isolate_islands(img, threshold = threshold)
        perim_pred = bwperim(isolated_pred)

        # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
        if mask is not None:
            mask_bool = mask < 0.5
            perim_pred[mask_bool] = 0

        # masking bwperim_output on the img boundaries as 0 
        perim_pred[0, :] = 0
        perim_pred[perim_pred.shape[0]-1, :] = 0
        perim_pred[:, 0] = 0
        perim_pred[:, perim_pred.shape[1]-1] = 0
    
        ellipse_info = gen_ellipse_contour_perim(perim_pred)
        return ellipse_info
    
    elif method=="Abd":
        #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        try:
            contours = sorted(contours, key=cv2.contourArea)
            pnts=contours[-1]
            
            pnts=np.int64(pnts[:,0,:])
            ellipse_info = FitEllipse_RobustLSQ(pnts, img, 5, 95.0)
            
            return ellipse_info
        except (IndexError,ValueError,ZeroDivisionError) :
            return None

def fit_ellipse_compact(img, threshold = 0.5, mask=None):
    """Fitting an ellipse to the thresholded pixels which form the largest connected area.

    Args:
        img (2D numpy array): Prediction from the DeepVOG network (240, 320), float [0,1]
        threshold (scalar): thresholding pixels for fitting an ellipse
        mask (2D numpy array): Prediction from DeepVOG-3D network for eyelid region (240, 320), float [0,1].
                                intended for masking away the eyelid such as the fitting is better
    Returns:
        ellipse_info (tuple): A tuple of (center, w, h, radian), center is a list [x-coordinate, y-coordinate] of the ellipse centre. 
                                None is returned if no ellipse can be found.
    """
    
        
    isolated_pred = isolate_islands(img, threshold = threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        mask_bool = mask < 0.5
        perim_pred[mask_bool] = 0

    # masking bwperim_output on the img boundaries as 0 
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0]-1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1]-1] = 0
    
    ellipse_info = gen_ellipse_contour_perim_compact(perim_pred)
    return ellipse_info
    
    

def FitEllipse_RobustLSQ(pnts, roi,  max_refines=5, max_perc_inliers=95.0):
    
    # Debug flag
    DEBUG = False

    # Suppress invalid values
    np.seterr(invalid='ignore')

    # Maximum normalized error squared for inliers
    max_norm_err_sq = 4.0

    # Tiny circle init
    best_ellipse = ((0,0),(1e-6,1e-6),0)

    # Count edge points
    n_pnts = pnts.shape[0]


    # Break if too few points to fit ellipse (RARE)
    if n_pnts < 5:
        return best_ellipse

    # Fit ellipse to points
    ellipse = cv2.fitEllipse(pnts)

    # Refine inliers iteratively
    for refine in range(0, max_refines):

        # Calculate normalized errors for all points
        norm_err = EllipseNormError(pnts, ellipse)

        # Identify inliers
        inliers = np.nonzero(norm_err**2 < max_norm_err_sq)[0]

        # Update inliers set
        inlier_pnts = pnts[inliers]

        # Protect ellipse fitting from too few points
        if inliers.size < 5:
            if DEBUG: print('Break < 5 Inliers (During Refine)')
            break

        # Fit ellipse to refined inlier set
        ellipse = cv2.fitEllipse(inlier_pnts)

        # Count inliers (n x 2)
        n_inliers    = inliers.size
        perc_inliers = (n_inliers * 100.0) / n_pnts

        # Update best ellipse
        best_ellipse = ellipse

        if perc_inliers > max_perc_inliers:
            if DEBUG: print('Break > maximum inlier percentage')
            break

    return best_ellipse

def EllipseNormError(pnts, ellipse):
    (x0,y0), (bb,aa), phi_b_deg = ellipse
    b = bb/2
    phi_b_rad = phi_b_deg * np.pi / 180.0
    bx, by = np.cos(phi_b_rad), np.sin(phi_b_rad)
    p1 = np.array( (x0 + (b + 1) * bx, y0 + (b + 1) * by) ).reshape(1,2)
    err_p1 = EllipseError(p1, ellipse)
    err_pnts = EllipseError(pnts, ellipse)
    return err_pnts / err_p1

def EllipseError(pnts, ellipse):
    np.seterr(divide='ignore')
    distance, grad, absgrad, normgrad = ConicFunctions(pnts, ellipse)
    err = distance / absgrad
    return err
def ConicFunctions(pnts, ellipse):
    np.seterr(invalid='ignore')
    conic = Geometric2Conic(ellipse)
    C = np.array(conic)
    x, y = pnts[:,0], pnts[:,1]
    X = np.array( ( x*x, x*y, y*y, x, y, np.ones_like(x) ) )
    distance = C.dot(X)
    Cg = np.array( ( (2*C[0], C[1], C[3]), (C[1], 2*C[2], C[4]) ) )
    Xg = np.array( (x, y, np.ones_like(x) ) )
    grad = Cg.dot(Xg)
    absgrad = np.sqrt(np.sqrt(grad[0,:]**2 + grad[1,:]**2))
    normgrad = grad / absgrad
    return distance, grad, absgrad, normgrad

def Geometric2Conic(ellipse):
    (x0,y0), (bb, aa), phi_b_deg = ellipse
    a, b = aa/2, bb/2
    phi_b_rad = phi_b_deg * np.pi / 180.0
    ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)
    a2 = a*a
    b2 = b*b
    if a2 > 0 and b2 > 0:
        A = ax*ax / a2 + ay*ay / b2;
        B = 2*ax*ay / a2 - 2*ax*ay / b2;
        C = ay*ay / a2 + ax*ax / b2;
        D = (-2*ax*ay*y0 - 2*ax*ax*x0) / a2 + (2*ax*ay*y0 - 2*ay*ay*x0) / b2;
        E = (-2*ax*ay*x0 - 2*ay*ay*y0) / a2 + (2*ax*ay*x0 - 2*ax*ax*y0) / b2;
        F = (2*ax*ay*x0*y0 + ax*ax*x0*x0 + ay*ay*y0*y0) / a2 + (-2*ax*ay*x0*y0 + ay*ay*x0*x0 + ax*ax*y0*y0) / b2 - 1;
    else:
        A,B,C,D,E,F = (1,0,1,0,0,-1e-6)
        
    conic = np.array((A,B,C,D,E,F))
    return conic