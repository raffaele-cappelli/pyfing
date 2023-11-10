##### Script to test both segmentation algorithms on FVC databases #####

import time
import json
import numpy as np

import sys
sys.path.append(".") # To import packages from this project
import pyfing as pf
from pyfing.segmentation import compute_segmentation_error, compute_dice_coefficient, compute_jaccard_coefficient
from common.fvc_segmentation_utils import load_fvc_db_and_gt

PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_PARAMS = './parameters/segmentation/'
PATH_RES = '../results/'

def compute_metrics(masks, gt):
    results = [(k // 8 + 1, (k % 8) + 1, 
                (100*compute_segmentation_error(m, x), 100*compute_dice_coefficient(m, x), 
                 100*compute_jaccard_coefficient(m, x))) for k, (m, x) in enumerate(zip(masks, gt))]
    metrics = np.array([x for _, _, x in results])
    return metrics, results

def run_test(alg: pf.SegmentationAlgorithm, year, db, subset):
    # load algorithm- and dataset-specific parameters
    alg.parameters = alg.parameters.load(f'{PATH_PARAMS}fvc{year}_db{db}_b_{type(alg).__name__}_params.json')
    images, gt = load_fvc_db_and_gt(PATH_FVC, PATH_GT, year, db, subset, 1, 100, 1, 8)
    start = time.time()
    masks = alg.run_on_db(images)
    elapsed = time.time() - start
    metrics, results = compute_metrics(masks, gt)
    avg_metrics = np.mean(metrics, 0)
    alg_name = type(alg).__name__
    print(f'Tested {alg_name} on FVC{year} DB{db}_{subset} ({len(images)} images). '\
          f'Err = {avg_metrics[0]:5.2f}% [{metrics[:,0].min():5.2f}%, {metrics[:,0].max():5.2f}%] '\
          f'DC = {avg_metrics[1]:5.2f} [{metrics[:,1].min():5.2f}, {metrics[:,1].max():5.2f}] '\
          f'JC = {avg_metrics[2]:5.2f} [{metrics[:,2].min():5.2f}, {metrics[:,2].max():5.2f}] '\
          f'Tot time: {elapsed:5.2f}s Avg: {elapsed/len(images):.4f}s')    
    with open(f'{PATH_RES}fvc{year}_db{db}_{subset}_{alg_name}_res.txt', 'w') as file:
        json.dump(results, file)    
    return avg_metrics

## 

TEST_DATASETS = [(y, db, "a") for y in (2000, 2002, 2004) for db in (1,2,3,4)]

print('Testing...')
for alg in [pf.Gmfs(), pf.Sufs()]:
    avg_er, avg_dc, avg_jc = np.mean([run_test(alg, y, db, subset) for y, db, subset in TEST_DATASETS], 0)
    print(f'Avg metrics: ER = {avg_er:.2f}% DC = {avg_dc:.2f} JC = {avg_jc:.2f}')
