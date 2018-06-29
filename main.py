from config import config, device
from preproc import preproc
from absl import app
import math
import os
import numpy as np
import ujson as json
import re
from collections import Counter
import string
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

'''
Some functions are from the official evaluation script.
'''
writer = SummaryWriter()

class SQuADDataset(Dataset):
    def __init__(self, npz_file, num_steps, batch_size, name='train'):
        data = np.load(npz_file)
        self.name = name
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()
        num = len(self.ids)
        self.batch_size = batch_size
        self.num_steps = num_steps if num_steps >= 0 else num // batch_size
        num_items = self.num_steps * batch_size
        idxs = list(range(num))
        self.idx_map = []
        i, j = 0, num

        while j <= num_items:
            random.shuffle(idxs)
            self.idx_map += idxs.copy()
            i = j
            j += num
        random.shuffle(idxs)
        self.idx_map += idxs[:num_items - i]

    def __len__(self):
        return self.num_steps
    def getname(self):
        return self.name

    def __getitem__(self, item):
        idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
        res = (self.context_idxs[idxs],
               self.context_char_idxs[idxs],
               self.ques_idxs[idxs],
               self.ques_char_idxs[idxs],
               self.y1s[idxs],
               self.y2s[idxs], self.ids[idxs])
        return res


def getans(p1, p2, length):
    max_ans_len = config.ans_limit
    ans_pro = float("inf")
    ret_st, ret_ed = 0, 0
    for i in range(length - max_ans_len):
        for j in range(max_ans_len):
            assert p1[i]<0 and p2[i+j]<0
            cur = -p1[i] * -p2[i+j]
            if cur < ans_pro:
                ans_pro = cur
                ret_st = i
                ret_ed = j
    return ret_st, ret_ed
    
def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        l = len(spans)
        start, end = getans(p1, p2, l)
        ans = context[spans[start][0]: spans[end][1]]
        answer_dict[str(qid)] = ans
        remapped_dict[uuid] = ans
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def update(model, optimizer, scheduler, data):
    model.train()
    optimizer.zero_grad()
    Cwid, Ccid, Qwid, Qcid, y1, y2, ids = data
    Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
    p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
    y1, y2 = y1.to(device), y2.to(device)
    loss = F.nll_loss(p1, y1) + F.nll_loss(p2, y2)
    ret = loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip)
    return ret

def valid(model, dataset, eval_file, num_ex=-1):
    print('start to valid {}'.format(dataset.getname()))
    model.eval()
    answer_dict = {}
    total_loss = 0 
    num_batches = num_ex//config.batch_size if num_ex>=0 else len(dataset)
    with torch.no_grad():
        for i in tqdm(random.sample(range(0, len(dataset)), num_batches), total=num_batches):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[i]
            Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(p1, y1) + F.nll_loss(p2, y2)
            total_loss += loss.item()
            answer_dict_, _ = convert_tokens(eval_file, ids.data.cpu().numpy(), 
                                            p1.data.cpu().numpy(), 
                                            p2.data.cpu().numpy())
            answer_dict.update(answer_dict_)
    metrics = evaluate(eval_file, answer_dict)
    return total_loss/num_batches, metrics
    '''
    return metrics, loss_count
    '''
    #print("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))

def train(config):
    from models import QANet

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    model = QANet(word_mat, char_mat).to(device)
    train_dataset = SQuADDataset(config.train_record_file, config.num_steps, config.batch_size)
    dev_dataset = SQuADDataset(config.dev_record_file, -1, config.batch_size, name='dev')

    lr = config.learning_rate
    base_lr = 1.0
    lr_warm_up_num = config.lr_warm_up_num

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2), eps=1e-7, weight_decay=3e-7, params=parameters)
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr)

    L = config.checkpoint
    N = config.num_steps
    best_f1 = 0
    best_em = 0
    patience = 0
    unused = True
    for iter in tqdm(range(0, N)):
        if iter % L == 0:
            valid_train_loss, valid_train_metrics = valid(model, train_dataset, train_eval_file, num_ex=1000)
            valid_dev_loss, valid_dev_metrics = valid(model, dev_dataset, dev_eval_file, num_ex=1000)
            if config.use_tensorboard:
                writer.add_scalar('data/valid_train_loss', valid_train_loss, iter/L)
                writer.add_scalar('data/valid_dev_loss', valid_dev_loss, iter/L)
                writer.add_scalar('data/valid_train_em', valid_train_metrics['exact_match'], iter/L)
                writer.add_scalar('data/valid_dev_em', valid_dev_metrics['exact_match'], iter/L)
                writer.add_scalar('data/valid_train_f1', valid_train_metrics['f1'], iter/L)
                writer.add_scalar('data/valid_dev_f1', valid_dev_metrics['f1'], iter/L)
        train_loss = update(model, optimizer, scheduler, train_dataset[iter])
        if config.use_tensorboard:
            writer.add_scalar('data/train_loss', train_loss, iter)
        if iter + L >= lr_warm_up_num - 1 and unused:
            optimizer.param_groups[0]['initial_lr'] = lr
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.decay)
            unused = False
        #print("Learning rate: {}".format(scheduler.get_lr()))
        '''
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        print('after {} steps , f1={} em={}'.format(iter, dev_f1, dev_em))
        if dev_em < best_em:
            patience += 1
            print('doesnot beat best model, patience={}'.format(patience))
            if patience > config.early_stop:
                break
        else:
            fn = os.path.join(config.save_dir, "model.pt")
            result = os.path.join(config.save_dir, "best_result.txt")
            print('beat best model, now best em={}, f1={}'.format(dev_em, dev_f1))
            with open(result, "w") as f:
                json.dump(metrics, f)
            best_em = dev_em
            best_f1 = max(best_f1, dev_f1)
            torch.save(model, fn)
            patience = 0
        '''
    writer.close()


def main(_):
    
    if config.disable_dropout:
        config.dropout = 0
        config.dropout_char = 0
    if config.mode == "train":
        train(config)
    elif config.mode == "data":
        preproc(config)
    elif config.mode == "debug":
        config.num_steps = 1000
        config.checkpoint = 10
        config.early_stop = 10000
        config.use_tensorboard = True
        train(config)
    elif config.mode == "test":
        test_entry(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)
