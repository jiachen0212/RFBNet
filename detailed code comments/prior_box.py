import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    Note:
    This 'layer' has changed between versions of the original SSD paper,
    so we include both versions, but note v2 is the most tested and most recent version of the paper.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']                # 输入RFBNet的图像尺度
        self.num_priors = len(cfg['aspect_ratios'])     # 各个feature map上预定义的anchor长宽比清单，与检测分支的数量对应
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']         # 特征金字塔层上各个feature map尺度
        self.min_sizes = cfg['min_sizes']               # 预定义的anchor尺度
        # 可以回想下，SSD中6个default bbox如何定义的？2:1 + 1:2 + 1:3 + 3:1
        # 两个1:1长宽比的anchor，但SSD定义了一个根号2尺度的anchor，max_sizes类似，但并不是严格对应的
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']                       # stride   其实是感受野吧

        # number of priors for feature map location (either 4 or 6)
        self.aspect_ratios = cfg['aspect_ratios']       # feature map上每个pix上预定义4 / 6个anchor，[2, 3]对应6个anchor
        self.clip = cfg['clip']                         # 位置校验
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []  # 保存所有feature map上预定义的anchor
        for k, f in enumerate(self.feature_maps):       # 对特征金字塔的各个检测分支，每个feature map上each-pixel都做密集anchor采样
            for i, j in product(range(f), repeat=2):    # 笛卡尔乘积，可以开始密集anchor采样了
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k                    # 以上三步操作，就相当于从feature map位置映射至原图，float型

                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]              # 第一个anchor添加，1:1长宽比

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]  # 第二个anchor添加，1:1长宽比，尺度与第一个anchor不一样，和SSD对应上了~~~

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    # aspect_ratios[k]:
                    # [2,3] layer1  4
                    # [2,3] layer2  4
                    # [2,3] layer3  4
                    # [2,3] layer4  4
                    # [2] layer5    2
                    # [2] layer6    2 anchors
                    # *sqrt(2) : 1/sqrt(2) == 2 : 1
                    # *sqrt(3) : 1/sqrt(3) == 3 : 1
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]   # 就是对应类似[2,3]，生成2:1 + 1:2 + 1:3 + 3:1四个长宽比的anchor了
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]   # 如果是[2]，就只有2:1 + 1:2两个长宽比，刚好与4 / 6个anchor对应

        # 总结：
        # 1 feature map上each-pixel对应4 / 6个anchor，长宽比：2:1 + 1:2 + 1:3 + 3:1 + 1:1 + 1:1，后两个1:1的anchor对应的尺度有差异；
        # 2 跟SSD还是严格对应的，每个feature map上anchor尺度唯一(2:1 + 1:2 + 1:3 + 3:1 + 1:1这五个anchor的尺度还是相等的，面积相等)，仅最后的1:1 anchor尺度大一点；
        # 3 所有feature map上所有预定义的不同尺度、长宽比的anchor保存至mean中；

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)    # 操作类似reshape，规则化输出
        if self.clip:
            output.clamp_(max=1, min=0)            # float型坐标校验
        return output
