import cv2 as cv


fvc_db_non_500_dpi = {(2002,2): 569, (2004,3): 512} # FVC datasets with non-standard DPI

def load_fvc_db_and_gt(path_db, path_gt, year, db, subset, finger_from=None, finger_to=None, impression_from = 1, impression_to = 8):
    if finger_from is None and finger_to is None:
        i1, i2 = (1, 100) if subset=="a" else (101, 110)
    else:
        i1, i2 = finger_from, finger_to
    j1, j2 = impression_from, impression_to
    return ([cv.imread(f'{path_db}fvc{year}/db{db}_{subset}/{i}_{j}.png', cv.IMREAD_GRAYSCALE)
            for i in range(i1, i2+1) for j in range(j1, j2+1)],
            [cv.bitwise_not(cv.imread(f'{path_gt}fvc{year}_db{db}_im_{i}_{j}seg.png', 
            cv.IMREAD_GRAYSCALE)) for i in range(i1, i2+1) for j in range(j1, j2+1)])
