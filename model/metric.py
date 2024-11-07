import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def kappa(output, target):
    pass


def weighted_kappa(output, target, weight=2):
    # metric simple explanation
    # https://datatab.net/tutorial/weighted-cohens-kappa
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        k = pred.shape[1]

        assert pred.shape[0] == len(target)

        # observed frequencies
        fo = torch.zeros(k, k, dtype=torch.int64)
        for t, p in zip(target.view(-1), pred.view(-1)):
            fo[t.long(), p.long()] += 1

        # weight matrix
        w = torch.zeros((k,k))
        for i in range(k):
            for j in range(k):
                w[i, j] = (i - j)**weight / (k -1)**weight

        # expected frequencies
        total = fo.sum()
        fe = (fo.sum(0)/total * fo.sum(1).unsqueeze(1)/total)*total

        kw = 1 - (w*fo).sum()/(w*fe).sum()

    return kw
    