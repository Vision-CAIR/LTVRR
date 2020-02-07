import torch.nn as nn
from utils.memory_utils import *

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path='Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_1_gpu/gvqa/Feb07-10-55-03_login104-09_step_with_prd_cls_v3/ckpt/model_step1439.pth',
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
