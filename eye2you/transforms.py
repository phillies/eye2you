import torchvision


class SlidingWindowCrop():

    def __init__(self, size, stride=1):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        if isinstance(size, (tuple, list)):
            self.size = size
        self.stride = stride

    def __call_(self, pic):
        w, h = pic.size
        p_h, p_w = self.size
        n_h = int((h - p_h) / self.stride)
        n_w = int((h - p_h) / self.stride)
        patches = []
        for ii in range(n_h):
            for jj in range(n_w):
                patch = torchvision.transforms.functional.crop(pic, ii * self.stride, jj * self.stride, p_h, p_w)
                patches.append(patch)

        return patches

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, stride={1})'.format(self.size, self.stride)