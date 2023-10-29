from dataclasses import dataclass
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt

def extract_row_col_numbers(boundingBoxes):
    starts_x = [b.x for b in boundingBoxes]
    n, bins = np.histogram(starts_x, bins=200)
    col_ranges = [(bins[i], bins[i+1]) for i,x in enumerate(n) if x > 0]

    starts_y = [b.y for b in boundingBoxes]
    n, bins = np.histogram(starts_y, bins=200)
    row_ranges = [(bins[i], bins[i+1]) for i,x in enumerate(n) if x > 0]


    for box in boundingBoxes:
        col_number = None
        row_number = None

        for i, point in enumerate(col_ranges):
            if point[0] <= box.x <= point[1]:
                box.col = i
                break
        for i, point in enumerate(row_ranges):
            if point[0] <= box.y <= point[1]:
                box.row = i
                break
    
    return boundingBoxes

@dataclass
class BoxAnnotation:
    # this is a common data structure we use in many projects, pls use it as it is.
    x: int  # upper left corner x (absolute)
    y: int  # upper left corner y (absolute)
    width: int  # box width (absolute)
    height: int  # box height (absolute)
    class_name: str  # identifier for row and column starting at 0 (format: f"cell-{row}-{col}")
    text: str
    row: int
    col: int

class LineItem():
    def __init__(self, item):
        x1, y1 = item[1][0], item[1][1]
        x2, y2 = item[2][0], item[2][1]
        self.width = abs(x1 - x2)
        self.height = abs(y1 - y2)

def sort_contours(cnts, method="left-to-right"):
    
    # construct the list of bounding boxes
    boundingBoxes = []
    for c in cnts:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        ### find out how "boxy" the contours are


        # Calculate the extent of the contour
        contour_area = cv2.contourArea(c)
        bounding_rect_area = w * h
        extent = float(contour_area) / bounding_rect_area

        if extent > 0.9: 
            boundingBoxes.append(BoxAnnotation(x,y,w,h, None, None, None, None)) 

    # initialize the reverse flag and sort index
    reverse = False
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
      
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                            key=lambda b:b[1].y, reverse=reverse))
       
    # handle if we are sorting against the x-coordinate
    else:
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                    key=lambda b:b[1].x, reverse=reverse))
        
    # return the list of sorted contours and bounding boxes




    
    # there can be some relicts. We want to remove them if they're significantly smaller than the mean.
    # # there's also some bounding boxes bigger than the cells (if there's an annotation over more cells. We exclude those)
    # boundingBoxes = [box for box in boundingBoxes if box.width > 0.5*hmean]    
    # boundingBoxes = [box for box in boundingBoxes if box.height > 0.15*vmean] 
    
    # hmean = np.mean([box.height for box in boundingBoxes]) 
    # vmean = np.mean([box.width for box in boundingBoxes])
    boundingBoxes = [box for box in boundingBoxes if box.width > 4]    
    boundingBoxes = [box for box in boundingBoxes if box.height > 4]

    boundingBoxes = [box for box in boundingBoxes if box.x != 0]    
    boundingBoxes = [box for box in boundingBoxes if box.y != 0]

    # find the bbox with the greatest area to remove it (better (TODO): if it's weight spans approximately all others)
    max_area = 0    
    max_bbox = None
    for bbox in boundingBoxes:
        # Calculate the area of the current bounding box
        area = bbox.width * bbox.height

        # Check if the current area is greater than the maximum area found so far
        if area > max_area:
            max_area = area
            max_bbox = bbox

    # Remove the biggest bounding box from the list
    
    boundingBoxes = list(boundingBoxes)

    if boundingBoxes:
        boundingBoxes.remove(max_bbox)


    boundingBoxes = extract_row_col_numbers(boundingBoxes[1:]) ##### very ugly. If we correcty extracted the table, then the first one is the table sized bounding box. So skipping 
 
    # sort first by row, then by column: 
    boundingBoxes = sorted(boundingBoxes, key=lambda box: (box.row, box.col))    
    return (cnts, boundingBoxes) 

def define_kernels(img):
    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return (ver_kernel, hor_kernel, kernel)

def extract_bounding_boxes(img, split_columns = False):
        
    #converting to binary
    _, img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)

    #inverting the image 
    img_bin = ~img_bin
    
    # define kernels based on image size
    ver_kernel, hor_kernel, kernel = define_kernels(img)

    # Use vertical kernel to detect vertical lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    # dilating increases visibility of lines 
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3) 

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # weigh both images equally when weighting them
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    
    # threshold can be as low as we want it to be as long as its greater than 0
    _, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # contour: (x,y) coordinates of points on a line making up the object
    
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort all the contours by left to right, create their bounding boxes
    _, boundingBoxes = sort_contours(contours, method= "top-to-bottom")
    
    # flatten output
    for box in boundingBoxes:
        cropped_image = img_bin[box.y:box.y+box.height, box.x:box.x+box.width]
        # print(img_bin.shape)

        text = pytesseract.image_to_string(cropped_image)
        box.text = text

    return boundingBoxes