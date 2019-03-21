import numpy as np
from numba import jit
import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return izip(a, b)

def poses2boxes(poses):
    global seen_bodyparts
    """
    Parameters
    ----------
    poses: ndarray of human 2D poses [People * BodyPart]
    Returns
    ----------
    boxes: ndarray of containing boxes [People * [x1,y1,x2,y2]]
    """
    boxes = []
    for person in poses:
        seen_bodyparts = person[np.where((person[:,0] != 0) | (person[:,1] != 0))]
        # box = [ int(min(seen_bodyparts[:,0])),int(min(seen_bodyparts[:,1])),
        #        int(max(seen_bodyparts[:,0])),int(max(seen_bodyparts[:,1]))]
        mean = np.mean(seen_bodyparts, axis=0)
        deviation = np.std(seen_bodyparts, axis = 0)
        box = [int(mean[0]-deviation[0]), int(mean[1]-deviation[1]), int(mean[0]+deviation[0]), int(mean[1]+deviation[1])]
        boxes.append(box)
    return np.array(boxes)

def distancia_midpoints(mid1, mid2):
    return np.linalg.norm(np.array(mid1)-np.array(mid2))

def pose2midpoint(pose):
    """
    Parameters
    ----------
    poses: ndarray of human 2D pose [BodyPart]
    Returns
    ----------
    boxes: pose midpint [x,y]
    """
    box = poses2boxes([pose])[0]
    midpoint = [np.mean([box[0],box[2]]), np.mean([box[1],box[3]])]
    return np.array(midpoint)

@jit
def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)
