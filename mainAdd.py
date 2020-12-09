import torch
import argparse
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('mister_ed'))

# ReColorAdv
from reid import models
from torch.nn import functional as F
import os.path as osp
from reid import datasets
from reid.utils.data import transforms as T
from torchvision.transforms import Resize
from reid.utils.data.preprocessor import Preprocessor
from reid.evaluators import Evaluator
from torch.optim.optimizer import Optimizer, required
import numpy as np
import random
import math
from reid.evaluators import extract_features
import faiss
from reid.utils.meters import AverageMeter
import torchvision

CHECK = 1e-5
# CHECK = 1e-3
SAT_MIN = 0.5
MODE = "bilinear"


class ResNet(models.ResNet):
    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        x2 = self.feat(x1)
        x2 = self.feat_bn(x2)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x2 = self.classifier_x2(x2)
        return x1, x2


def get_data(sourceName, mteName, targetName, split_id, data_dir, height, width,
             batch_size, workers, combine_trainval):
    root = osp.join(data_dir, sourceName)
    rootMte = osp.join(data_dir, mteName)
    rootTgt = osp.join(data_dir, targetName)
    sourceSet = datasets.create(sourceName, root, num_val=0.1, split_id=split_id)
    mteSet = datasets.create(mteName, rootMte, num_val=0.1, split_id=split_id)
    tgtSet = datasets.create(targetName, rootTgt, num_val=0.1, split_id=split_id)
    num_classes = (sourceSet.num_trainval_ids if combine_trainval else sourceSet.num_train_ids)
    class_tgt = (tgtSet.num_trainval_ids if combine_trainval else tgtSet.num_train_ids)

    train_transformer = T.Compose([
        Resize((height, width)),
        T.ToTensor(),
    ])

    # generate dataset
    curSet = [(osp.join(sourceSet.images_dir, val[0]), val[1], val[2]) for val in sourceSet.trainval]

    for val in mteSet.trainval:
        curSet.append((osp.join(mteSet.images_dir, val[0]), num_classes + int(val[1]), int(val[2])))
    meta_train = DataLoader(
        Preprocessor(curSet, root=None, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return sourceSet, tgtSet, mteSet, num_classes, class_tgt, meta_train


def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min


class MI_SGD(Optimizer):
    def __init__(
            self,
            params,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            max_eps=10 / 255,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MI_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["sign"]:
                    d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group["sign"]:
                    p.data.add_(-group["lr"], d_p.sign())
                    p.data = torch.clamp(p.data, -self.max_eps, self.max_eps)
                else:
                    p.data.add_(-group["lr"], d_p)

        return loss


def train(train_loader, net, noise, epoch, optimizer, centroids, normalize):
    global args
    noise.requires_grad = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mean = torch.Tensor(normalize.mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(normalize.std).view(1, 3, 1, 1).cuda()

    net.eval()

    end = time.time()
    optimizer.zero_grad()
    optimizer.rescale()
    for i, (input, _, _, _) in enumerate(train_loader):
        # measure data loading time.
        data_time.update(time.time() - end)
        model.zero_grad()
        input = input.cuda()
        with torch.no_grad():
            norm_output = (input - mean) / std
            feature = net(norm_output)[0]
            scores = centroids.mm(F.normalize(feature.t(), p=2, dim=0))
            realLab = scores.max(0, keepdim=True)[1]
            _, ranks = torch.sort(scores, dim=0, descending=True)
            pos_i = ranks[0, :]
            neg_i = ranks[-1, :]
        neg_feature = centroids[neg_i, :].view(-1, 2048)  # centroids--512*2048
        pos_feature = centroids[pos_i, :].view(-1, 2048)

        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE, size=tuple(input.shape[-2:]),
            align_corners=True,
        ).squeeze()
        perturted_input = torch.clamp(input + current_noise, 0, 1)
        perturted_input_norm = (perturted_input - mean) / std
        perturbed_feature = net(perturted_input_norm)[0]

        optimizer.zero_grad()
        pair_loss = 10 * F.triplet_margin_loss(perturbed_feature, neg_feature, pos_feature, 0.5)

        fakePred = centroids.mm(perturbed_feature.t()).t()
        oneHotReal = torch.zeros(scores.t().shape).cuda()
        oneHotReal.scatter_(1, realLab.view(-1, 1), float(1))
        label_loss = F.relu(
            (fakePred * oneHotReal).sum(1).mean() - (fakePred * (1 - oneHotReal)).max(1)[0].mean()
        )

        pair_loss = pair_loss.view(1)

        loss = pair_loss + label_loss

        loss.backward()

        losses.update(loss.item())
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                ">> Train: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Noise l2: {noise:.4f}".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    noise=noise.norm(),
                )
            )

    noise.requires_grad = False
    print(f"Train {epoch}: Loss: {losses.avg}")
    return losses.avg, noise


def calDist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def test(dataset, net, noise, args, evaluator, epoch, saveRank=False, saveCount=5):
    print(">> Evaluating network on test datasets...")

    net = net.cuda()
    net.eval()
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def add_noise(img):
        n = noise.cpu()
        img = img.cpu()
        n = F.interpolate(
            n.unsqueeze(0), mode=MODE, size=tuple(img.shape[-2:]), align_corners=True
        ).squeeze()
        return torch.clamp(img + n, 0, 1)

    query_trans = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(), T.Lambda(lambda img: add_noise(img)),
        normalize
    ])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(), normalize
    ])
    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.images_dir, transform=query_trans),
        batch_size=args.batch_size, num_workers=0, shuffle=False, pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True
    )
    qFeats, gFeats, testQImage, qnames, gnames = [], [], [], [], []
    with torch.no_grad():
        for (inputs, qname, _, _) in query_loader:
            inputs = inputs.cuda()
            qFeats.append(net(inputs)[0])
            if len(testQImage) < saveCount:
                testQImage.append(inputs[0, ...])
            qnames.extend(qname)
        qFeats = torch.cat(qFeats, 0)
        for (inputs, gname, _, _) in gallery_loader:
            inputs = inputs.cuda()
            gFeats.append(net(inputs)[0])
            gnames.extend(gname)
        gFeats = torch.cat(gFeats, 0)
    distMat = calDist(qFeats, gFeats)
    if saveRank:
        import cv2
        # plot rakning list of 0001
        _, ind = torch.sort(distMat, 1)
        ind = ind[0, :8]
        results = [osp.join(gallery_loader.dataset.root, gnames[val]) for val in ind]
        allNames = [osp.join(query_loader.dataset.root, qnames[0])] + results
        isCorr = [1 if int(qnames[0].split('/')[-1].split('_')[0]) == int(val.split('/')[-1].split('_')[0])
                  else 0 for val in results]
        # imshow
        ranklist = []
        for ii, (name, mask) in enumerate(zip(allNames, isCorr)):
            img = cv2.imread(name)
            if ii != 0:
                img = cv2.rectangle(img, (0, 0), (64, 128),
                                    (0, 255, 0) if mask == 1 else (0, 0, 255), 2)
            ranklist.append(img)
            if ii == 0:
                ranklist.append(np.zeros((128, 20, 3)))
        ranklist = np.concatenate(ranklist, 1)
        cv2.imwrite(f'{epoch}.jpg', ranklist)

    # evaluate on test datasets
    evaluator.evaMat(distMat, dataset.query, dataset.gallery)
    return testQImage


def save_noise(noise, epoch, image, idx):
    filename = os.path.join(args.noise_path, f"noise_{epoch}_{idx}")
    noiseName = os.path.join(args.noise_path, f"noise_{epoch}_addnoise_{idx}")
    torchvision.utils.save_image(noise, filename + ".png", normalize=True)
    torchvision.utils.save_image(image, noiseName + ".png", normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a ResNet-50 trained on Imagenet '
                    'against ReColorAdv'
    )

    parser.add_argument('--data', type=str, required=True,
                        help='path to reid dataset')
    parser.add_argument('--noise_path', type=str,
                        default='.', help='path to reid dataset')
    parser.add_argument('-s', '--source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-t', '--target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-m', '--mte', type=str, default='personx',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=50, required=True,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--resumeTgt', type=str, default='', metavar='PATH')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument("--max-eps", default=8, type=int, help="max eps")
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    sourceSet, tgtSet, mteSet, num_classes, class_tgt, mix_loader = \
        get_data(args.source, args.mte, args.target,
                 args.split, args.data, args.height,
                 args.width, args.batch_size, 8, args.combine_trainval)

    model = ResNet(50, pretrained=True, num_classes=num_classes)
    modelTest = ResNet(50, pretrained=True, num_classes=class_tgt)
    if args.resume:
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        try:
            model.load_state_dict(checkpoint)
        except:
            allNames = list(checkpoint.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkpoint[name]
            model.load_state_dict(checkpoint, strict=False)
        checkTgt = torch.load(args.resumeTgt)
        if 'state_dict' in checkTgt.keys():
            checkTgt = checkTgt['state_dict']
        try:
            modelTest.load_state_dict(checkTgt)
        except:
            allNames = list(checkTgt.keys())
            for name in allNames:
                if name.count('classifier') != 0:
                    del checkTgt[name]
            modelTest.load_state_dict(checkTgt, strict=False)

    model.eval()
    modelTest.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        modelTest = modelTest.cuda()

    features, _ = extract_features(model, mix_loader, print_freq=10)
    features = torch.stack([features[f] for f, _, _ in mix_loader.dataset.dataset])

    ncentroids = 512
    fDim = features.shape[1]
    cluster = faiss.Kmeans(fDim, ncentroids, niter=20, gpu=True)
    cluster.train(features.cpu().numpy())
    centroids = torch.from_numpy(cluster.centroids).cuda().float()

    evaluator = Evaluator(modelTest, args.print_freq)
    evaSrc = Evaluator(model, args.print_freq)

    # universal noise
    noise = torch.zeros((3, args.height, args.width)).cuda()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    noise.requires_grad = True
    MAX_EPS = args.max_eps / 255.0

    optimizer = MI_SGD(
        [
            {"params": [noise], "lr": MAX_EPS / 10, "momentum": 1, "sign": True}
        ],
        max_eps=MAX_EPS,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))

    # train on training set, add noise on query only
    import time

    for epoch in range(args.epoch):
        scheduler.step()
        begin_time = time.time()
        loss, noise = train(
            mix_loader, model, noise, epoch, optimizer, centroids, normalize
        )
        if epoch % 5 == 4:
            testQImage = test(tgtSet, modelTest, noise, args, evaluator, epoch)
            # testQ = test(sourceSet, model, noise, args, evaSrc, epoch)
            for ii, curImg in enumerate(testQImage):
                save_noise(noise, epoch, curImg, ii)