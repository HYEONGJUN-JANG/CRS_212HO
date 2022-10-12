import os
from datetime import datetime
from pytz import timezone

from parameters import parse_args
from main import main
import numpy as np

NUM_TRIAL = 5
content_hits, initial_hits, best_results = [], [], []
if __name__ == '__main__':
    args = parse_args()
    command = 'python main.py'
    for i, v in vars(args).items():
        # print(i, v)
        command += f' --{i}={v}'
    # num_metric = 0
    for _ in range(NUM_TRIAL):
        content_hit, initial_hit, best_result = main(args)
        content_hits.append(content_hit)
        initial_hits.append(initial_hit)
        best_results.append(best_result)

    print(content_hits)

    avg_content_hits = np.mean(np.array(content_hits), axis=0)
    avg_initial_hits = np.mean(np.array(initial_hits), axis=0)
    avg_best_results = np.mean(np.array(best_results), axis=0)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))  # MonthDailyHourMinute .....e.g., 05091040
    results_file_path = f"./results/Final_{mdhm}_name_{args.name}.txt"
    # parameters
    with open(results_file_path, 'w', encoding='utf-8') as result_f:
        result_f.write(
            '\n=================================================\n')
        for i, v in vars(args).items():
            result_f.write(f'{i}:{v} || ')
        result_f.write('\n')

        result_f.write('content_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_content_hits) + '\n')
        result_f.write('initial_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_initial_hits) + '\n')
        result_f.write('best_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_best_results) + '\n')
