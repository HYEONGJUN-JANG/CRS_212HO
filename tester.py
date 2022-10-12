import os
from datetime import datetime
from pytz import timezone

from parameters import parse_args
from main import main
import numpy as np

NUM_TRIAL = 1
all_results = []
if __name__ == '__main__':
    args = parse_args()
    command = 'python main.py'
    for i, v in vars(args).items():
        # print(i, v)
        command += f' --{i}={v}'
    # num_metric = 0
    for _ in range(NUM_TRIAL):
        result = main(args)
        all_results.append(result)
    all_results = np.array(all_results)
    print(all_results)
    avg_result = np.mean(all_results, axis=0)
    print(avg_result)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))  # MonthDailyHourMinute .....e.g., 05091040
    results_file_path = f"./results/Final_{mdhm}_name_{args.name}.txt"
    # parameters
    with open(results_file_path, 'w', encoding='utf-8') as result_f:
        result_f.write('\t'.join(map(str, list(avg_result))))
