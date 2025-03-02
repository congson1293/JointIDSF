import logging
import os

import numpy as np
import torch
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

from sklearn.metrics import accuracy_score, f1_score


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        # self.config = self.config_class.from_pretrained(model_path, finetuning_task=args.task)

        if args.pretrained:
            print(args.model_name_or_path)
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
        # GPU or CPU
        try:
            torch.cuda.set_device(self.args.gpu_id)
        except:
            pass
        # print(self.args.gpu_id)
        # print(torch.cuda.current_device())
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        # print("check init")
        # results = self.evaluate("dev")
        # print(results)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch", _)

            total_intent_accuracy, total_slot_f1 = 0, 0
            total_loss = 0

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "intent_label_ids": batch[3],
                    "slot_labels_ids": batch[4],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]
                intent_logits, slot_logits = outputs[1]

                acc = self.cal_intent_acc(batch[3], intent_logits)
                total_intent_accuracy += acc

                f1 = self.cal_slot_f1(batch[4], batch[1], slot_logits)
                total_slot_f1 += f1

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                total_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            avg_intent_acc = total_intent_accuracy / len(train_dataloader)
            avg_slot_f1 = total_slot_f1 / len(train_dataloader)
            avg_loss = total_loss / len(train_dataloader)

            print(f'loss = {avg_loss}\ti_acc={avg_intent_acc}\te_f1={avg_slot_f1}')

            train_iterator.set_postfix(loss=avg_loss, i_acc=avg_intent_acc, e_f1=avg_slot_f1)

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
            writer.add_scalar("Loss/train", tr_loss / global_step, _)

        self.save_model()

        return global_step, tr_loss / global_step

    def cal_intent_acc(self, labels, preds):
        intent_logits = torch.argmax(preds, dim=1).detach().cpu().numpy()
        intent_label_ids = labels.detach().cpu().numpy()
        acc = accuracy_score(intent_label_ids, intent_logits)
        return acc

    def cal_slot_f1(self, slots, slots_mask, slot_logit):
        total_f1 = 0
        preds = self.model.crf.decode(slot_logit, mask=slots_mask.bool())
        slots_ = slots.detach().cpu().numpy()
        for i in range(len(slots_)):
            pred = preds[i]
            slot = slots_[i][:len(pred)]
            f1 = f1_score(slot, pred, average='macro')
            total_f1 += f1
        return total_f1 / len(slots_)


    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "intent_label_ids": batch[3],
                    "slot_labels_ids": batch[4],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs["intent_label_ids"].detach().cpu().numpy(), axis=0
                )

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(
                    out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                args=self.args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")
