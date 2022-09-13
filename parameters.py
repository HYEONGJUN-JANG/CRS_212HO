import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # 프로그램 실행 시 필수로 넘겨줘야하는 인자들
    parser.add_argument('--name', type=str, default='plot-review')  # plot or review or plot-review
    parser.add_argument('--n_sample', type=int, default=3)  # 1 or 2 or 3 or 20
    # parser.add_argument('--n_num', type=int, default=3) # 1 or 2 or 3 or 20
    parser.add_argument('--max_plot_len', type=int, default=100)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--max_review_len', type=int, default=100)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--max_dialog_len', type=int, default=100)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--kg_emb_dim', type=int, default=128)
    parser.add_argument('--num_bases', type=int, default=8)

    parser.add_argument('--max_title_len', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='valid_portion')

    parser.add_argument('--bert_name', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'albert-base-v2', 'prajjwal1/bert-small', 'prajjwal1/bert-mini'])

    parser.add_argument('--pretrained', action='store_false')
    parser.add_argument('--test', action='store_false')

    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        pass # train_loader = DataLoader(redial_train, batch_size=256, shuffle=True)  # HJ KT-server
    elif sysChecker() == "Windows":
        pass # train_loader = DataLoader(redial_train, batch_size=4, shuffle=True)  # HJ local
    else:
        print("Check Your Platform and use right DataLoader")
        exit()
    logging.info(args)
    return args


# main
if __name__ == "__main__":
    args = parse_args()
