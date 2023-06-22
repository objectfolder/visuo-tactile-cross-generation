import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_list = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.val_list.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count