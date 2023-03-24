import os
from datetime import datetime
from pytz import timezone

from parameters import parse_args
from main import main
import numpy as np

from utils import get_time_kst
from pygit2 import Repository
import sys

# NUM_TRIAL = 5
content_hits, initial_hits, best_results = [], [], []
if __name__ == '__main__':

    args = parse_args()
    command = 'python main.py --name=plot-review-serial-filepathtest'

    # Git branch
    repo = Repository(os.getcwd())
    branch = repo.head.shorthand
    hash = str(repo.head.target)

    # for i, v in sys.argv:
    #     # print(i, v)
    #     command += f' --{i}={v}'
    # num_metric = 0
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))  # MonthDailyHourMinute .....e.g., 05091040
    finalSubfolder_name = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d') + '_final')
    finalFolder_path = os.path.join('./results', finalSubfolder_name)
    if not os.path.exists(finalFolder_path): os.mkdir(finalFolder_path)

    # results_file_path = f"./results/Final_{mdhm}_name_{args.name}.txt"
    results_file_path = os.path.join(finalFolder_path, f"Final_{mdhm}_name_{args.name}.txt")
    with open(results_file_path, 'w', encoding='utf-8') as result_f:
        result_f.write('\n=================================================\n')
        result_f.write(get_time_kst())
        result_f.write('\n')
        result_f.write(branch)
        result_f.write('\n')
        result_f.write(hash)
        result_f.write('\n')
        result_f.write(str(sys.argv))
        result_f.write('\n')
        for i, v in vars(args).items():
            result_f.write(f'{i}:{v} || ')
        result_f.write('\n')

    if 'rec' in args.task:
        for t in range(args.num_trial):
            content_hit, initial_hit, best_result = main(args)
            content_hits.append(content_hit)
            initial_hits.append(initial_hit)
            best_results.append(best_result)
            with open(results_file_path, 'a', encoding='utf-8') as result_f:
                result_f.write('#TRIAL:\t%d\n' % t)
                result_f.write('content_hits:\t' + '\t'.join(format(x, ".2f") for x in content_hit) + '\n')
                result_f.write('initial_hits:\t' + '\t'.join(format(x, ".2f") for x in initial_hit) + '\n')
                result_f.write('best_hits:\t' + '\t'.join(format(x, ".2f") for x in best_result) + '\n')

        # print(content_hits)

        avg_content_hits = np.mean(np.array(content_hits), axis=0)
        avg_initial_hits = np.mean(np.array(initial_hits), axis=0)
        avg_best_results = np.mean(np.array(best_results), axis=0)

        # parameters
        with open(results_file_path, 'a', encoding='utf-8') as result_f:
            result_f.write('[AVERAGE]\n')
            result_f.write('content_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_content_hits) + '\n')
            result_f.write('initial_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_initial_hits) + '\n')
            result_f.write('best_hits:\t' + '\t'.join(format(x, ".2f") for x in avg_best_results) + '\n')
    elif 'conv' in args.task:
        for t in range(args.num_trial):
            main(args)