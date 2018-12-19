import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
    
    Usage: 
    am = AverageMeter()
    am.update(123)
    am.update(456)
    am.update(789)
    
    last_value = am.val
    average_value = am.avg
    
    am.reset() #set all to 0"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccuracyMeter(object):
    """Computes and stores the correctly classified samples
    
    Usage: pass the number of correct (val) and the total number (n)
    of samples to update(val, n). Then .avg contains the 
    percentage correct.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class AccuracyMeter2(object):
    """Computes and stores the correctly classified samples
    
    Usage: pass the number of correct (val) and the total number (n)
    of samples to update(val, n). Then .avg contains the 
    percentage correct.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.count_b = 0
        self.sum_b = 0
        self.avg_b = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.count_b = 0
        self.sum_b = 0
        self.avg_b = 0        

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_b(self, val, n):
        self.sum_b += val
        self.count_b += n
        self.avg_b = self.sum_b / self.count_b

class AccuracyMeterMulti(object):
    """Computes and stores the correctly classified samples
    
    Usage: pass the number of correct (val) and the total number (n)
    of samples to update(val, n). Then .avg contains the 
    percentage correct.
    """
    def __init__(self, n=1):
        self.avg = 0
        self.sum = 0
        self.count = np.zeros(n, dtype=np.int)

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = np.zeros_like(self.count)

    def update(self, val, n):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count