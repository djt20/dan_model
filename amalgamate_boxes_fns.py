'''
amalgamate boxes function
amalgamates overlapping bounding boxes and counts
inputs:
 - bounding box coords, growths counted (in the form [[[bb1_tl, bb1_br], bb1_count], [[bb2_tl, bb2_br], bb2_count]]
 - img size
outputs:
 - bounding box coords (amalgamated), growths counted
program flow:
 - create mask from original bounding box coords
 - create bounding boxes from edges of each connected area (cv2.findContours)
 - for each found contour:
    - compare to original bboxs
    - if original bbox in found contour:
        - add count to new bbox
 - also provide input and output masks w/ box coords + count for comparison
FUTURE improvements?
 - amalgamate on the basis of percentage overlap of original box/iou, not just any overlap
'''
import numpy as np
import cv2
import itertools

input_boxes = [[[55, 55], [105, 105], 1],
               [[110, 65],[170, 85], 1],
               [[100,100], [200,200], 2],
               [[150,150], [165, 165], 1],
               [[195, 195], [230, 210], 1],
               [[420, 420], [440, 440], 1]]


def modify_input_boxes(in_boxes):
    modified = []
    for box in in_boxes:
        box_mod = [[box[0], box[1]], box[2]]
        modified.append(box_mod)
    return modified


def create_binary_mask(test_boxes, size, show=False):
    binary_mask = np.zeros((size))

    for box in test_boxes:
        tl = box[0][0]
        br = box[0][1]
        cv2.rectangle(binary_mask, tuple(tl), tuple(br), 255, -1)

    if show:
        cv2.imshow("test", binary_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return binary_mask


def find_bounding_boxes(binary_mask, contours):
    bb_out_mask = np.zeros_like(binary_mask)
    out_coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        tl = [x, y]
        br = [x+w, y+h]
        coords = [[tl, br]]
        cv2.rectangle(bb_out_mask, (x, y), (x + w, y + h), 255, -1)
        out_coords.append(coords)
    return out_coords


def box_enclosed(boundb, innerb):  # check if smaller box is enclosed by bigger box
    if boundb[0][0][0] <= innerb[0][0][0] and boundb[0][0][1] <= innerb[0][0][1]:
        # If bottom-right inner box corner is inside the bounding box
        if innerb[0][1][0] <= boundb[0][1][0] and innerb[0][1][1] <= boundb[0][1][1]:
            return True
        else:
            return False
    else:
        return False


def amalgamate(bboxs_with_counts):
    bboxs_with_counts = modify_input_boxes(bboxs_with_counts)
    bin_mask = create_binary_mask(bboxs_with_counts, (2048, 2048), False)
    cnts, hierarchy = cv2.findContours(bin_mask.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    amalg_box_coords = find_bounding_boxes(bin_mask, cnts)
    amalg_box_coords_w_counts = [bb.append(0) or bb for bb in amalg_box_coords]
    for a, b in itertools.combinations(amalg_box_coords_w_counts, 2):
        if box_enclosed(a, b):
            a[1] += b[1]
            amalg_box_coords_w_counts.remove(b)
        elif box_enclosed(b, a):
            b[1] += a[1]
            if a in amalg_box_coords_w_counts:
                amalg_box_coords_w_counts.remove(a)

    for bigbox in amalg_box_coords_w_counts:
        for smallbox in bboxs_with_counts:
            if box_enclosed(bigbox, smallbox):
                bigbox[1] += smallbox[1]

    out = []
    for box in amalg_box_coords_w_counts:
        box_out = [box[0][0], box[0][1], box[1], box[1]]
        out.append(box_out)

    return out