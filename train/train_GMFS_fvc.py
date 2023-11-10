##### Script to optimize the parameters of GMFS on FVC datasets B #####

import sys
import copy
import numpy as np

sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error
from common.fvc_segmentation_utils import load_db, load_gt, fvc_db_non_500_dpi


PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_RES = '../results/'

def average_err_on_db(images, gt, parameters):
    """
    Computes the average segmentation error on a database
    """
    alg = pf.Gmfs(parameters)
    masks = alg.run_on_db(images)
    errors = [compute_segmentation_error(m, x) for m, x in zip(masks, gt)]
    return np.mean(errors)


def optimize_parameters_on_db(year, db, subset, images, gt):
    """
    Looks for the best combination of parameters among some sets of reasonable values
    """
    dpi = fvc_db_non_500_dpi.get((year, db), 500)
    p = pf.GmfsParameters(0, 0, 0, 1, 2, dpi)
    min_err, best_parameters = 100, p
    for p.percentile in [95]: # We use a fixed 95-percentile
        for p.sigma in [7/3, 9/3, 11/3, 13/3, 15/3, 17/3, 19/3, 21/3, 23/3, 25/3]: # A reasonable range of sigma values
            for p.threshold in np.arange(0.02, 0.28, 0.01): # A reasonable range of thr values
                min_err, best_parameters = search_step(year, db, subset, images, gt, min_err, best_parameters, p)

    p.percentile = best_parameters.percentile
    p.sigma = best_parameters.sigma
    p.threshold = best_parameters.threshold
    for p.closing_count in [2, 3, 4, 5, 6, 8, 9]: # A few resonable values for cc
        for p.opening_count in [6, 9, 12, 15, 18, 21]: # A few resonable values for oc
            min_err, best_parameters = search_step(year, db, subset, images, gt, min_err, best_parameters, p)

    # saves best parameters to file
    best_parameters.save(f'{PATH_RES}fvc{year}_db{db}_b_{pf.Gmfs.__name__}_params.json')


def search_step(year, db, subset, images, gt, min_err, best_parameters, parameters):
    e = average_err_on_db(images, gt, parameters)
    if e < min_err:
        min_err, best_parameters = e, copy.copy(parameters)
    print(f'{year}-{db}-{subset}: {parameters} -> {e:.2f}% (best: {min_err:.2f}%)')
    return min_err, best_parameters


def main():
    """
    Optimizes the parameters on the FVC B datasets
    """
    for y, db, subset in [(y, db, "b") for y in (2000, 2002, 2004) for db in (1,2,3,4)]:
        images = load_db(PATH_FVC, y, db, subset)
        gt = load_gt(PATH_GT, y, db, subset)
        optimize_parameters_on_db(y, db, subset, images, gt)


main()
