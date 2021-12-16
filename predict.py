import os

import numpy as np
import torch
from torch.nn.functional import softmax

from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, load_tokenizer


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(model_dir):
    return torch.load(os.path.join(model_dir, "training_args.bin"))


def load_model(args, device):
    # Check whether model exists
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            args.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


class JointIDSF:
    def __init__(self):
        self.model_dir = 'models'
        self.device = 'cpu'
        self.args = get_args(self.model_dir)
        self.model = load_model(self.args, self.device)
        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        self.pad_token_label_id = self.args.ignore_index
        self.tokenizer = load_tokenizer(self.args)

    def build_tensor_words(self, words, args, tokenizer, pad_token_label_id, cls_token_segment_id=0,
                           pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True, device='cpu'):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
        slot_label_mask = torch.tensor([slot_label_mask], dtype=torch.long).to(device)

        return input_ids, attention_mask, token_type_ids, slot_label_mask

    def predict(self, text):
        words = text.split()

        input_ids, attention_mask, token_type_ids, slot_label_mask = self.build_tensor_words(words, self.args,
                                                                                             self.tokenizer,
                                                                                             self.pad_token_label_id,
                                                                                             device=self.device)

        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "intent_label_ids": None,
                "slot_labels_ids": None,
            }
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = token_type_ids
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            intent_logits = softmax(intent_logits, dim=1)
            intent_preds = intent_logits.detach().cpu().numpy()[0]
            top_k_intents = intent_preds.argsort()[-5:][::-1]
            top_k_intents_score = intent_preds[top_k_intents]
            top_k_intents_label = [self.intent_label_lst[i] for i in top_k_intents]

            # Slot prediction
            slot_preds = np.array(self.model.crf.decode(slot_logits))[0][1:]
            slot_label_mask = slot_label_mask.detach().cpu().numpy()[0][1:]
            slot_label = []
            for i in range(len(slot_label_mask)):
                mask = slot_label_mask[i]
                if mask == self.pad_token_label_id:
                    continue
                label = self.slot_label_lst[slot_preds[i]]
                slot_label.append(label)
            slots = self.extract_label(words, slot_label)

        intent_result, top_k_intent_result = self.convert_intent_to_rasa_format(top_k_intents,
                                                                                top_k_intents_label,
                                                                                top_k_intents_score)
        slot_result = self.convert_slot_to_rasa_format(slots)

        return intent_result, top_k_intent_result, slot_result

    def extract_label(self, words, slot_label):
        slots = {}
        tag, entity, flag = None, [], False
        processed_words = []
        for i in range(len(slot_label)):
            word, label = words[i], slot_label[i]
            if label == 'O':
                if flag:
                    self.update_dict(slots, tag, entity)
                    flag = False
            elif 'B_' in label:
                if flag:
                    self.update_dict(slots, tag, entity)
                tag = label.replace('B_', '')
                entity = [word]
                flag = True
            else:  # 'I_' case
                entity.append(word)
            processed_words.append(word)
        if flag:
            self.update_dict(slots, tag, entity)

        return slots

    @staticmethod
    def update_dict(d: dict, key: str, val: list):
        try:
            v = ' '.join(val)
            d[key].append(v)
        except:
            d[key] = [v]

    def convert_intent_to_rasa_format(self, top_k_intents_id, top_k_intents_label, top_k_intents_score):
        result = []
        for i in range(len(top_k_intents_score)):
            intent_id = top_k_intents_id[i]
            intent_name = top_k_intents_label[i]
            intent_score = top_k_intents_score[i]
            result.append({'id': intent_id,
                          'name': intent_name,
                          'confidence': intent_score})
        return result[0], result

    def convert_slot_to_rasa_format(self, slots):
        result = []
        for k, v in slots.items():
            for e in v:
                result.append({'entity': k,
                               'value': e,
                               'confidence_entity': 0.5,
                               'extractor': 'JointIDSF'})
        return result


if __name__ == "__main__":
    j = JointIDSF()
    j.predict('mở tài khoản')
