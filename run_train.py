import argparse

from src.train_handler import TrainHandler

#### ArgParse for Model details
model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')

group = model_parser.add_mutually_exclusive_group(required=True)
group.add_argument('--exp_name', type=str,         help='name to save the experiment as')
group.add_argument('--temp', action='store_true',  help='to save in temp dir', )

model_parser.add_argument('--transformer', default='longformer', type=str,  help='select transformer (e.g. bert, roberta etc.)')
model_parser.add_argument('--encoder',     default='utt_trans',  type=str,  help='select encoder')
model_parser.add_argument('--decoder',     default='linear',     type=str,  help='select decoder')
model_parser.add_argument('--system_args', default=None,         type=str,  help='select system arguments',   nargs='+')

model_parser.add_argument('--num_labels',  default=43,       type=int,  help='')
model_parser.add_argument('--device',      default='cuda',   type=str,  help='device to use (cuda, cpu)')

model_parser.add_argument('--formatting', default=None, type=str, help='formatting input ids')
model_parser.add_argument('--max_len',    default=4090,      type=int, help='training print size for logging')
model_parser.add_argument('--filters',    default=None,      type=str, help='text filters (e.g. punctuation)', nargs='+')
model_parser.add_argument('--num_seeds',  default=1,         type=int,  help='number of seeds to train')

#### ArgParse for Training details
train_parser = argparse.ArgumentParser(description='Arguments for training the system')

tr_path = f"swda/standard/train.json"
dev_path = f"swda/standard/dev.json"
test_path = f"swda/standard/test.json"

train_parser.add_argument('--train_path', default=tr_path,   type=str,  help='')
train_parser.add_argument('--dev_path',   default=dev_path,  type=str,  help='')
train_parser.add_argument('--test_path',  default=test_path, type=str,  help='')
train_parser.add_argument('--lim',     default=None, type=int, help='size of data subset to use (for debugging)')
train_parser.add_argument('--print_len', default=100,   type=int,  help='logging training print size')

train_parser.add_argument('--epochs',  default=10,     type=int,   help='numer of epochs to train')
train_parser.add_argument('--lr',      default=1e-5,  type=float, help='training learning rate')
train_parser.add_argument('--bsz',     default=4,     type=int,   help='training batch size')

train_parser.add_argument('--optim',   default='adamw', type=str, help='which optimizer to use (adam, adamw)')
train_parser.add_argument('--sched',   default=None,   type=str,  help='which scheduler to use (triangle, exponential, step)')
train_parser.add_argument('--s_args',  default=None,   type=list, help='scheduler arguments to use (depends on scheduler)')

if __name__ == '__main__':
    model_args = model_parser.parse_known_args()[0]
    train_args = train_parser.parse_known_args()[0]

    if model_args.system_args:  assert all([i in ['spkr_embed', 'utt_embed'] for i in model_args.system_args])
    if model_args.filters:      assert all([i in ['punctuation', 'action', 'hesitation'] for i in model_args.filters])  

    if model_args.num_seeds == 1:
        trainer = TrainHandler(model_args.exp_name, model_args)
        trainer.train(train_args)
    else:
        for i in range(model_args.num_seeds):
            exp_name = model_args.exp_name + '/' + str(i)
            trainer = TrainHandler(exp_name, model_args)
            trainer.train(train_args)
