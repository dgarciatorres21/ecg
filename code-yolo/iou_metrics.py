import torch
import math

# calculate iou
def box_iou(box1, box2):
    # get the coordinates of the intersection rectangle
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y3 = torch.min(box1[3], box2[3])

    # compute the area of intersection
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y3 - y1, min=0)

    # compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the area of union
    union_area = box1_area + box2_area - inter_area
    
    # compute the iou
    iou = inter_area / (union_area + 1e-6) # add epsilon to avoid division by zero
    return iou

# calculate giou
def generalized_box_iou(box1, box2):

    iou = box_iou(box1, box2)

    # find the coordinates of the smallest enclosing box
    c_x1 = torch.min(box1[0], box2[0])
    c_y1 = torch.min(box1[1], box2[1])
    c_x2 = torch.max(box1[2], box2[2])
    c_y2 = torch.max(box1[3], box2[3])

    # area of the enclosing box
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    inter_area = iou * (box1_area + box2_area - iou * (box1_area + box2_area)) # simplified
    union_area = box1_area + box2_area - inter_area

    # giou
    giou = iou - (c_area - union_area) / (c_area + 1e-6)
    return giou

# calculate ciou
def complete_box_iou(box1, box2):
    iou = box_iou(box1, box2)

    # smallest enclosing box
    c_x1, c_y1 = torch.min(box1[0], box2[0]), torch.min(box1[1], box2[1])
    c_x2, c_y2 = torch.max(box1[2], box2[2]), torch.max(box1[3], box2[3])
    
    # diagonal of enclosing box
    c_diag = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    # center points of boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    
    center1_x, center1_y = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    center2_x, center2_y = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    
    # distance between center points
    center_dist = (center1_x - center2_x)**2 + (center1_y - center2_y)**2

    # diou component
    diou_term = center_dist / (c_diag + 1e-6)
    diou = iou - diou_term

    # ciou component (aspect ratio consistency)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    
    ciou = iou - (diou_term + alpha * v)
    
    # return all values for analysis
    return iou, giou, diou, ciou
