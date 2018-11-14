import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode

# test_RFB.py中被调用，生成的detector对象，结合net输出的out（forward中拆分为loc + conf，结合RFBNet结构，
# 可以发现是一个全卷积的feature map，可以结合multibox函数理解），conf就是预测的分类得分，
# loc其实就是预定义位置、尺度、长宽比的anchor（也称为prior box）的offsets，再结合anchor + loc offsets，就可以得到最终预测的结果了
# 至于之后的NMS、top-k阈值抑制，都是后操作，不属于Detect里做的工作
class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc, conf = predictions    # test_RFB.py内调用net，通过RFBNet输出的(loc, conf)

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)                                            # batch size
        self.num_priors = prior_data.size(0)                              # 预定义的anchor个数，如SSD，指的是特征金字塔上所有检测分支feature map上的anchor
        self.boxes = torch.zeros(1, self.num_priors, 4)                   # 对应bbox(x1, y1, x2, y2)，batch size = 1
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)   # 对应bbox类别，如VOC 20 + 1类，batch size = 1
        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)             # batch size = 1，维度规整化
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)   # 应该是num_priors -> self.num_priors，因为有batch张图像，所以做了一个类似reshape的操作
            self.boxes.expand_(num, self.num_priors, 4)     # batch size = num
            self.scores.expand_(num, self.num_priors, self.num_classes) # batch size = num

        # Decode predictions into bboxes.
        for i in range(num):   # for each detected image
            # Decode locations from predictions using priors to undo the encoding we did for offset regression
            # decode函数功能很容易理解，结合RCNN中的supplements，一下就能明白了
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()   # 只需view操作即可，然后就直接赋值咯

            self.boxes[i] = decoded_boxes    # (x1, x2, y1, y2)
            self.scores[i] = conf_scores     # batch内每个图像对应bbox、score

        return self.boxes, self.scores