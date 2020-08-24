import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CharRNN train & generate')
    subparsers = parser.add_subparsers(dest='func', help='train or generate')

    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.add_argument('--input_file', dest='input_file', required=True, help='bar help')
    parser_train.add_argument('--save_file', dest='save_file', required=True, help='bar help')
    parser_train.add_argument('--num_seqs', dest='num_seqs', type=int, default=32, help='bar help')
    parser_train.add_argument('--time_steps', dest='time_steps', type=int, default=50, help='bar help')
    parser_train.add_argument('--lstm_units', dest='lstm_units', type=int, default=128, help='bar help')
    parser_train.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='bar help')
    parser_train.add_argument('--use_embedding', dest='use_embedding', action='store_true', help='bar help')
    parser_train.add_argument('--embedding_size', dest='embedding_size', type=int, default=128, help='bar help')
    parser_train.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='bar help')
    parser_train.add_argument('--droput_rate', dest='droput_rate', type=float, default=0.5, help='bar help')
    parser_train.add_argument('--max_vocab', dest='max_vocab', type=int, default=3500, help='bar help')
    parser_train.add_argument('--min_vocab_counting', dest='min_vocab_counting', type=int, default=10, help='bar help')

    parser_generate = subparsers.add_parser('generate', help='generate help')
    parser_generate.add_argument('--input_file', dest='input_file', required=True, help='bar help')
    parser_generate.add_argument('--save_file', dest='save_file', required=True, help='bar help')
    parser_generate.add_argument('--lstm_units', dest='lstm_units', type=int, default=128, help='bar help')
    parser_generate.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='bar help')
    parser_generate.add_argument('--embedding_size', dest='embedding_size', type=int, default=128, help='bar help')
    parser_generate.add_argument('--droput_rate', dest='droput_rate', type=float, default=0, help='bar help')
    parser_generate.add_argument('--start_string', dest='start_string', default='', help='bar help')
    parser_generate.add_argument('--length_to_generate', dest='length_to_generate', type=int, default=30, help='bar help')

    args = parser.parse_args()
    print(args)
    raise
