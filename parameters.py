import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # 프로그램 실행 시 필수로 넘겨줘야하는 인자들
    parser.add_argument('--name', type=str, default='plot-review')  # plot-serial or review-serial or plot-review or plot-review-serial or none
    parser.add_argument('--n_review', type=int, default=3)  # 1 or 2 or 3 or 20
    parser.add_argument('--n_plot', type=int, default=3) # 1 or 2 or 3 or 9
    parser.add_argument('--max_plot_len', type=int, default=50)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--max_review_len', type=int, default=50)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--max_dialog_len', type=int, default=50)  # 50, 100, 150, 200, 250, (300)
    parser.add_argument('--kg_emb_dim', type=int, default=128) #128
    parser.add_argument('--num_bases', type=int, default=8)

    parser.add_argument('--max_title_len', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--lr_pt', type=float, default=1e-5, help='Pre-training Learning rate')
    parser.add_argument('--lr_ft', type=float, default=1e-3, help='Fine-tuning Learning rate')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='valid_portion')
    parser.add_argument('--loss_lambda', type=float, default=0.1, help='lambda')
    parser.add_argument('--n_sample', type=int, default=3, help='sampling')

    # TransformerEncoder Args
    parser.add_argument('--word_encoder', type=float, default=1, help='0: bert, 1: transformer')
    parser.add_argument('--n_heads', type=int, default=2, help='n_heads')
    parser.add_argument('--n_layers', type=int, default=2, help='n_layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.0, help='attention_dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu_dropout')
    parser.add_argument('--learn_positional_embeddings', type=bool, default=False, help='learn_positional_embeddings')
    parser.add_argument('--embeddings_scale', type=bool, default=True, help='embeddings_scale')
    parser.add_argument('--ffn_size', type=int, default=300, help='ffn_size')
    parser.add_argument('--reduction', type=bool, default=False, help='reduction')
    parser.add_argument('--n_positions', type=int, default=1024, help='n_positions')
    parser.add_argument('--pad_token_id', type=int, default=0, help='pad_token_id')
    parser.add_argument('--vocab_size', type=int, default=30522, help='vocab_size')

    parser.add_argument('--bert_name', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'albert-base-v2', 'prajjwal1/bert-small', 'prajjwal1/bert-mini'])

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--test', action='store_false')

    args = parser.parse_args()

    logging.info(args)
    return args


# main
if __name__ == "__main__":
    args = parse_args()
