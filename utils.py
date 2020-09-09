'''
helper for holding simple custom functions i reuse loads
'''
import numpy as np
import cv2
from scipy import ndimage as nd

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
lineType = 2


def threshold_array(arr):
    arr[arr <= 125] = 0
    arr[arr > 125] = 255
    return arr


''' helper for localisation '''
def mask_to_bboxs(mask, msk):    # transforms any generic mask (growth patch/overlap to bboxs for each segment detected
    mod = 0
    if msk == "bg":
        mod = 2

    masked_gp = np.ma.masked_where(mask <= 100, mask)
    thresh = 101
    masked_gp = cv2.threshold(masked_gp, thresh, 255, cv2.THRESH_BINARY)[1]
    _, preds_gp = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    # find co-ords of non black areas
    labels, numL = nd.label(masked_gp)
    label_indices = [(labels == i).nonzero() for i in range(1, numL + 1)]
    all_coords = []
    bboxs = []
    n = 0
    for_rectangles = []
    for indices in label_indices:
        n += 1
        coords = zip(indices[0], indices[1])
        min_y = int(min(indices[0])) - mod
        max_y = int(max(indices[0])) + mod
        min_x = int(min(indices[1])) - mod
        max_x = int(max(indices[1])) + mod
        tl = [int(min_x), int(min_y)]
        br = [int(max_x), int(max_y)]
        bl = [int(min_x), int(max_y)]
        height = max_y - min_y
        width = max_x - min_x
        bbox = [tl, br]
        for_rectangles.append([bl, height, width])
        bboxs.append(bbox)
        all_coords.append(coords)
    return bboxs


''' display bboxs given as a list on an image, also outputs only requisite size bboxs'''
def display_bboxs_on_image(img_arr, bboxs, colour, minArea=8):
    img_arr = np.array(img_arr)
    bboxs_output = []
    for bb in bboxs:
        h = bb[0][0] - bb[1][0]
        w = bb[0][1] - bb[1][1]
        lb = bb[2]
        ub = bb[3]
        if lb == ub: count = str(lb)
        else: count = str(lb)+"-"+str(ub)
        area = h*w
        if area > minArea:
            cv2.rectangle(img_arr, tuple(bb[0]), tuple(bb[1]), colour, thickness=1)
            cv2.putText(img_arr, count, tuple(bb[0]), font, fontScale, colour, lineType)
            bboxs_output.append(bb)
    return img_arr, bboxs_output


''' checks what bounding boxes in bboxs_bigger fully contain one of the bboxs_smaller
    if smaller is in bigger, append final value of bb_smaller list to bigger, if not appends 1  '''
def check_smaller_in_bigger(bboxs_bigger, bboxs_smaller, mode):
    # Below: check if it is fully contained by a growth patch box
    for bb_smaller in bboxs_smaller:  # for every overlap box created
        tl_smaller, br_smaller = bb_smaller[0], bb_smaller[1]  # pull top left and btm right coords
        for bb_bigger in bboxs_bigger:  # compare to every growthPatch box
            tl_bigger, br_bigger = bb_bigger[0], bb_bigger[1]  # pull their top left and btm right coords
            if tl_smaller[0] - tl_bigger[0] >= 0 and br_bigger[0] - br_smaller[0] >= 0:  # if bb_gp fully contains an overlap box
                if tl_smaller[1] - tl_bigger[1] >= 0 and br_bigger[1] - br_smaller[1] >= 0:  # in both x and y
                    bb_bigger.append(bb_smaller[2])  # append "-1" to growthPatch box
                else:
                    if mode == "overlaps":
                        bb_bigger.append(1)
            else:
                if mode == "overlaps":
                    bb_bigger.append(1)
    return bboxs_bigger


def model_inference(img_arr, seg_model):                        # img_arr and seg_model to masks
    img_arr_expanded = np.expand_dims(img_arr / 255, axis=0)    # correct format for inference and normalise
    prediction = seg_model.predict(img_arr_expanded) * 255      # make predictions and unnormalise
    prediction = prediction[0, :, :, :]
    mask_growth, mask_overlaps, mask_background = prediction[:, :, 0], prediction[:, :, 1], prediction[:, :, 2]
    return mask_growth, mask_overlaps, mask_background


def count_growths_no_erosion(bboxs_bg, bboxs_gp, bboxs_ov):
    '''
    for all bounding boxes in background we return 1 growth, unless an overlap mask is also contained within it (==-1)
    '''
    bboxs_ov_info = []      # append -1 to end of each overlap bbox for future computation
    for bb_o in bboxs_ov:
        bb_o.append(-1)
        bboxs_ov_info.append(bb_o)  # output: each bb_overlap = [tl, br, -1]
    bboxs_gp_info = check_smaller_in_bigger(bboxs_gp, bboxs_ov, mode="overlaps")
    bboxs_gp_info_output = []
    for bb_gp in bboxs_gp_info:
        if -1 in bb_gp:
            bb_gp = [bb_gp[0], bb_gp[1], -1]
            bboxs_gp_info_output.append(bb_gp)
        elif len(bb_gp) == 2:
            bb_gp.append(1)
            bboxs_gp_info_output.append(bb_gp)
        elif len(bb_gp) >= 3:
            bb_gp = bb_gp[:3]       # output: each bb_gp = [tl, br, numGrowths] where numGrowths = -1 or 1
            bboxs_gp_info_output.append(bb_gp)
        else: raise Exception("How are we here?")

    bboxs_bg_info = check_smaller_in_bigger(bboxs_bg, bboxs_gp_info_output, mode="bg")
    bboxs_bg_info_output = []
    for bb_bg in bboxs_bg_info:
        if -1 in bb_bg:
            bb_bg = [bb_bg[0], bb_bg[1], -1, -1]
            bboxs_bg_info_output.append(bb_bg)
        else:
            count = sum(bb_bg[2:])
            if count == 0:
                count = 1
            bb_bg = [bb_bg[0], bb_bg[1], count, count]
            bboxs_bg_info_output.append(bb_bg)
    return bboxs_bg_info_output       # structure is list of bboxs with each entry = [tl, br, count, count]


def count_growths_morph_erosion(bboxs_bg, bboxs_gp, bboxs_ov):
    bboxs_ov_info = []      # append -1 to end of each overlap bbox for future computation
    for bb_o in bboxs_ov:
        bb_o.append(-1)
        bboxs_ov_info.append(bb_o)  # output: each bb_overlap = [tl, br, -1]


def plot_bboxs_gt(img_arr, reg_of_int): # todo: need to combine consumed (fully contained) bounding boxes
    bboxs = []
    for topleft, bottomright, lowerBound, upperBound in \
            zip(reg_of_int['tl'], reg_of_int['br'], reg_of_int['lowerBound'], reg_of_int['upperBound']):
        topleft, bottomright = topleft.split(" "), bottomright.split(" ")
        topleft = [int(x) for x in topleft]
        bottomright = [int(y) for y in bottomright]
        bboxs.append([topleft, bottomright, int(lowerBound), int(upperBound)])
    img_gt_bboxs, gt_bboxs = display_bboxs_on_image(img_arr, bboxs, (255,0,0))
    return img_gt_bboxs, gt_bboxs