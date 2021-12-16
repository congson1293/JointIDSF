import os

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import MODEL_PATH_MAP, init_logger, load_tokenizer, set_seed


class Args:
    def __init__(self, no_cuda=True, data_dir='VPS_data', model_dir='VPS_models'):
        self.root_dir = os.path.dirname(os.path.realpath(__file__))

        self.adam_epsilon = 1e-8
        self.attention_embedding_size = 100
        self.model_dir = os.path.join(self.root_dir, model_dir)
        self.data_dir = os.path.join(self.root_dir, data_dir)
        self.do_eval = False
        self.do_eval_dev = False
        self.do_train = True
        self.dropout_rate = 0.1
        self.early_stopping = 5
        self.embedding_type = 'soft'
        self.eval_batch_size = 64
        self.gpu_id = 0
        self.gradient_accumulation_steps = 1
        self.ignore_index = 0
        self.intent_label_file = 'intent_label.txt'
        self.intent_loss_coef = 0.5
        self.learning_rate = 5e-5
        self.logging_steps = 200
        self.max_grad_norm = 1.0
        self.max_seq_len = 30
        self.max_steps = -1
        self.model_type = 'phobert'
        self.no_cuda = no_cuda
        self.num_train_epochs = 100
        self.pretrained = False
        self.pretrained_path = os.path.join(self.root_dir, "./viatis_xlmr_crf")
        self.save_steps = 200
        self.seed = 1
        self.slot_label_file = "slot_label.txt"
        self.slot_pad_label = 'PAD'
        self.token_level = 'word-level'
        self.train_batch_size = 8
        self.tuning_metric = 'loss'
        self.use_attention_mask = False
        self.use_crf = True
        self.use_intent_context_attention = False
        self.use_intent_context_concat = False
        self.warmup_steps = 0
        self.weight_decay = 0.0

        self.model_name_or_path = MODEL_PATH_MAP[self.model_type]


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    if args.do_eval:
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    else:
        dev_dataset = None
    if args.do_eval_dev:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    else:
        test_dataset = None

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
    if args.do_eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")


def do_train(no_cuda=True, data_dir='VPS_data', model_dir='VPS_models'):
    args = Args(no_cuda=no_cuda, data_dir=data_dir, model_dir=model_dir)

    main(args)


if __name__ == "__main__":
    do_train()
