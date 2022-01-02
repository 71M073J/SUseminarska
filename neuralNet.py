import torch.nn as nn
import torch
import torch.optim
from nltk import PorterStemmer
from nltk.corpus import stopwords
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import pickle
import time

ps = PorterStemmer()
from datetime import datetime
from sentiment import get_pair_sentiment, savePairSent, get_sopojavitve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    def __init__(self, n_pairs):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(n_pairs, max(1, n_pairs // 4))
        self.fc2 = nn.Linear(max(1, n_pairs // 4), max(1, n_pairs // 16))
        self.fc3 = nn.Linear(max(1, n_pairs // 16), max(1, n_pairs // 64))
        self.fc4 = nn.Linear(max(1, n_pairs // 64), max(1, n_pairs // 64))
        self.fc5 = nn.Linear(max(1, n_pairs // 64), 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


class TrainData(Dataset):

    def __init__(self, dirn, train=True, seed=None, save_data=False, npairs=256, need_save=True):
        super(TrainData, self).__init__()
        if seed is None:
            seed = np.random.randint(1234567890)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.npairs = npairs
        data = []
        trening = "train" if train else "test"
        if save_data:
            stop_words = set(stopwords.words('english'))
            stop_words = stop_words.union(['-', 'br', "i'm", "he'", "i'v", "&"])

            data = np.zeros((25000, self.npairs + 1))

            with open(f"pairSentiment{npairs}.pkl", "rb") as f:
                vectordict = pickle.load(f)
            #print(vectordict)
            # vectordict = get_pair_sentiment()
            for attrib, senti in enumerate(["neg", "pos"]):
                # print(dirn, senti)
                dire = dirn + senti + "/"
                print("collecting data:", dire)
                for i, file in enumerate(os.listdir(dire)):
                    with open(dire + file, 'r', encoding="utf8") as f:
                        scores = {k: 0 for k in vectordict}
                        # print(scores)
                        for line in f:
                            line = line.replace("<br />", "").translate({ord(c): " " for c in './<>,\\\"()!?'}).lower()
                            col = [ps.stem(word) for word in line.split() if word not in stop_words]
                            le = len(col)
                            for i2 in range(le):
                                for i3 in range(i2 + 1, min(i2 + 20, le)):
                                    stem1 = col[i2]
                                    stem2 = col[i3]
                                    distscore = (i3 - i2)
                                    if stem1 < stem2:
                                        if (stem1, stem2) in scores:
                                            scores[(stem1, stem2)] += 1 / distscore
                                    else:
                                        if (stem2, stem1) in scores:
                                            scores[(stem2, stem1)] += 1 / distscore
                        debug = [scores[x] for x in scores] + [attrib]  # debug
                        data[attrib * 12500 + i] = debug
                    if i % 1000 == 0:
                        print(i)
            print("data collected, shuffling...")
        else:
            with open(f"fastLoadData{trening}{npairs}.pkl", "rb") as f:
                data = pickle.load(f)
        if save_data and need_save:
            with open(f"fastLoadData{trening}{npairs}.pkl", "wb") as f:
                pickle.dump(data, f)

        rng.shuffle(data)
        print("data shuffled.")
        self.dataXY = data.astype(np.float32)

    def __getitem__(self, index):
        return self.dataXY[index, :self.npairs], self.dataXY[index, self.npairs]

    def __len__(self):
        return len(self.dataXY)


def main(dvana=8):
    # dvana = 8 # 13 is 8192
    npairs = 2 ** dvana
    pathToCheckpoint = "model-22_09_40acc86.pt"
    checkpoint = False
    if dvana > 13:
        need_save = False
    else:
        need_save = True
    get_sopojavitve()
    savePairSent(npairs)
    # return
    EPOCHS = 69
    BATCH_SIZE = 64
    LEARNING_RATE = 0
    LEARNING_RATE = 0.05 / (2 ** dvana)
    model = NeuralNet(npairs)
    model.to(device)
    print(model)
    if checkpoint:
        model.load_state_dict(torch.load(pathToCheckpoint, device))
    lastdiff = 1
    lastAcc = 0
    vacc = 0
    factor = 0.001
    maxDiff = 0.99
    diff = 0
    currLearn = LEARNING_RATE
    save = True
    if os.path.exists(f"fastLoadDatatest{npairs}.pkl") and os.path.exists(f"fastLoadDatatrain{npairs}.pkl"):
        save = False
    seed = 1234
    train_data = TrainData("aclImdb_v1/aclImdb/train/", train=True, npairs=npairs, seed=seed,
                           save_data=save, need_save=need_save)
    valid_data = TrainData("aclImdb_v1/aclImdb/test/", train=False, npairs=npairs, seed=seed,
                           save_data=save, need_save=need_save)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)
    print("loader loaded...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    maxAcc = 0
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    for e in range(1, EPOCHS + 1):

        print(f"epoch {e} begins!")
        epoch_loss = 0
        epoch_acc = 0
        accs = []
        duration = time.time()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # torch.set_printoptions(profile="full")
            # print(X_batch)  # prints the whole tensor
            # torch.set_printoptions(profile="default")
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            accs.append(acc.item())
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f"Epoch time:{time.time() - duration}s")
        print(
            f'Epoch {e + 0:03}: |            Loss: {epoch_loss / len(train_loader):.5f} |'
            f' Acc: {epoch_acc / len(train_loader):.3f},'
            f' Mean/stderr: {torch.std_mean(torch.tensor(accs))}')
        if e % 1 == 0:
            model.eval()
            valid_loss = 0
            valid_acc = 0
            accs = []
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # torch.set_printoptions(profile="full")
                # print(X_batch)  # prints the whole tensor
                # torch.set_printoptions(profile="default")

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                accs.append(acc.item())

                valid_loss += loss.item()
                valid_acc += acc.item()
            lastAcc = vacc
            vacc = valid_acc / len(train_loader)
            # if vacc < 53:
            #    optimizer = optim.Adam(model.parameters(), lr=currLearn * 10)
            #    currLearn = currLearn * 10
            lastdiff = diff
            diff = lastAcc / vacc
            print("diff score:", diff)
            if vacc > 80 and diff > maxDiff:
                print("lowering learning rate...")
                optimizer = optim.Adam(model.parameters(), lr=currLearn // 10)
                currLearn = currLearn // 10
                maxDiff /= maxDiff + factor
                factor /= 10
            text = f'Epoch {e + 0:03}: | Validation Loss: {valid_loss / len(train_loader):.5f} |' + \
                   f' Acc: {vacc:.3f},' + \
                   f' stderr/mean: {torch.std_mean(torch.tensor(accs))}'
            print(text)
            print("diff, lastdiff, vacc, lastAcc:", diff, lastdiff, vacc, lastAcc)
            with open(f"model{npairs}report.txt", "a") as f:
                f.write(text + "\n")
            if vacc > maxAcc:
                maxAcc = vacc
                if maxAcc > 75:
                    print("saving model...")
                    torch.save(model.state_dict(), f"model{npairs}-acc{str(round(vacc))}.pt")
            if vacc > 70 and (diff > 1 and lastdiff > 1) or (maxAcc > vacc > 55 and lastAcc < maxAcc):
                print("Maximal valid acc already reached.")
                break
            model.train()


if __name__ == "__main__":
    for i in range(5, 15):
        print("running at i =", i)
        main(i)
