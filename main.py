import argparse
from model import CharRnn
from text import TextTokenizer

TEXT_FILE = "./data/poetry.txt"
SAVE_FILE = "./save/char_rnn.h5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CharRNN train & generate')
    subparsers = parser.add_subparsers(dest='func', help='train or generate')

    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.add_argument('--input_file', dest='input_file', default=TEXT_FILE, help='bar help')
    parser_train.add_argument('--save_file', dest='save_file', default=SAVE_FILE, help='bar help')
    parser_train.add_argument('--num_seqs', dest='num_seqs', type=int, default=32, help='bar help')
    parser_train.add_argument('--time_steps', dest='time_steps', type=int, default=50, help='bar help')
    parser_train.add_argument('--lstm_units', dest='lstm_units', type=int, default=128, help='bar help')
    parser_train.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='bar help')
    parser_train.add_argument('--use_embedding', dest='use_embedding', action='store_true', help='bar help')
    parser_train.add_argument('--embedding_size', dest='embedding_size', type=int, default=128, help='bar help')
    parser_train.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='bar help')
    parser_train.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.5, help='bar help')
    parser_train.add_argument('--max_vocab', dest='max_vocab', type=int, default=3500, help='bar help')
    parser_train.add_argument('--min_vocab_counting', dest='min_vocab_counting', type=int, help='bar help')
    parser_train.add_argument('--steps_per_epoch', dest='steps_per_epoch', type=int, default=2000, help='bar help')
    parser_train.add_argument('--epochs', dest='epochs', type=int, default=10, help='bar help')

    parser_generate = subparsers.add_parser('generate', help='generate help')
    parser_generate.add_argument('--input_file', dest='input_file', default=TEXT_FILE, help='bar help')
    parser_generate.add_argument('--save_file', dest='save_file', default=SAVE_FILE, help='bar help')
    parser_generate.add_argument('--num_seqs', dest='num_seqs', type=int, default=1, help='bar help')
    parser_generate.add_argument('--time_steps', dest='time_steps', type=int, default=1, help='bar help')
    parser_generate.add_argument('--lstm_units', dest='lstm_units', type=int, default=128, help='bar help')
    parser_generate.add_argument('--num_layers', dest='num_layers', type=int, default=2, help='bar help')
    parser_generate.add_argument('--use_embedding', dest='use_embedding', action='store_true', help='bar help')
    parser_generate.add_argument('--embedding_size', dest='embedding_size', type=int, default=128, help='bar help')
    parser_generate.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='bar help')
    parser_generate.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0, help='bar help')
    parser_generate.add_argument('--max_vocab', dest='max_vocab', type=int, default=3500, help='bar help')
    parser_generate.add_argument('--min_vocab_counting', dest='min_vocab_counting', type=int, help='bar help')
    parser_generate.add_argument('--start_string', dest='start_string', default='花', help='bar help')
    parser_generate.add_argument('--length_to_generate', dest='length_to_generate', type=int, default=50, help='bar help')

    args = parser.parse_args()
    if args.func == 'train':
        tokenizer = TextTokenizer(args.input_file, max_vocab=args.max_vocab,
                                  min_counting=args.min_vocab_counting)
        generator = tokenizer.training_data_generator(num_seqs=args.num_seqs,
                                                      time_steps=args.time_steps)
        model = CharRnn(tokenizer.vocab_size,
                        num_seqs=args.num_seqs, time_steps=args.time_steps,
                        use_embedding=args.use_embedding, embedding_size=args.embedding_size,
                        lstm_units=args.lstm_units, num_layers=args.num_layers,
                        learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,)
                        #save_file=args.save_file)
        model.train(generator, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs,
                    save_file=args.save_file)
    elif args.func == 'generate':
        tokenizer = TextTokenizer(args.input_file, max_vocab=args.max_vocab,
                                  min_counting=args.min_vocab_counting)
        encoded_start_string = tokenizer.encode(args.start_string)
        model = CharRnn(tokenizer.vocab_size,
                        num_seqs=args.num_seqs, time_steps=args.time_steps,
                        use_embedding=args.use_embedding, embedding_size=args.embedding_size,
                        lstm_units=args.lstm_units, num_layers=args.num_layers,
                        learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
                        save_file=args.save_file)
        sample = model.generate(encoded_start_string, args.length_to_generate)
        print("".join(tokenizer.decode(sample)))
