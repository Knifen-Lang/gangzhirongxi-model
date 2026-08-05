# Loss functions for job classification
from .cwbs_loss import CWBSLoss
from .focal_loss import FocalLoss, asymmetric_label_smoothing, cross_entropy_with_asymmetric_ls
from .contrastive_loss import SupConLoss, SpanContrastiveLoss
from .triplet_loss import TripletMarginLoss
from .desc_align_loss import DescAlignLoss
from .rdrop_loss import RDropLoss
from .large_margin_loss import LargeMarginLoss, LargeMarginLossV2
