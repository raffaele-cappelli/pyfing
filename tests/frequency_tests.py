##### Script to test both frequency estimation algorithms on the FFE benchmark #####
import time
import pyfing as pf
from pyfing.utils.ffe import load_ffe_dataset, compute_average_error_on_db


for alg in [pf.Xsffe(), pf.Snffe()]:
    print("Loading FFE datasets...")
    _db_ffe_good, _db_ffe_bad = load_ffe_dataset('../datasets/FFE', 'Good'), load_ffe_dataset('../datasets/FFE', 'Bad')
    print("Testing...")
    for stft_downsampling in [False, True]:
        start = time.time()
        eg_foe = compute_average_error_on_db(alg, _db_ffe_good, stft_downsampling)
        eb_foe = compute_average_error_on_db(alg, _db_ffe_bad, stft_downsampling)
        elapsed = time.time() - start
        print(f"Tested {type(alg).__name__} on FFE (STFT downsampling: {stft_downsampling}). MAPE: {eg_foe:.2f}% / {eb_foe:.2f}% Tot time: {elapsed:5.2f}s Avg: {elapsed/(len(_db_ffe_bad)+len(_db_ffe_good)):.4f}s")

