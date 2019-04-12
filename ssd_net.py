import torch
import torch.nn as nn
import torch.nn.functional as F
import util.module_util as module_util
from mobilenet import MobileNet


class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        # Setup the backbone network (base_net)
        self.base_net = MobileNet(num_classes)
        # The feature map will extracted from layer[11] and layer[13] in (base_net)
        self.base_output_layer_indices = (11, 13)
        # Define the Additional feature extractor
        self.additional_feat_extractor = nn.ModuleList([
            # Conv8_2
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Conv9_2
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            # Conv10_2
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ),
            # Conv11_2
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.ReLU(),
            ),
        ])

        # Bounding box offset regressor
        num_prior_bbox = 6                                                               # num of prior bounding boxes
        self.loc_regressor = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1), #Cov5_3
            nn.Conv2d(in_channels=1024, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1), #FC7
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1), #Conv8_2
            # TODO: implement remaining layers.
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),  #Conv9_2
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),  #Conv10_2
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * 4, kernel_size=3, padding=1),  #Conv11_2
        ])

        # Bounding box classification confidence for each label
        self.classifier = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=num_prior_bbox * num_classes, kernel_size=3, padding=1),
        ])

        # Load the pre-trained model for self.base_net, it will increase the accuracy by fine-tuning
        basenet_state = torch.load('pretrained/mobienetv2.pth', map_location='cpu')
        base_net_1 = {key: value for key, value in basenet_state.items() if 'base_net' in key}
        self.base_net.load_state_dict(base_net_1)
        layer_idx = 0

        def init_with_xavier(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        self.loc_regressor.apply(init_with_xavier)
        self.classifier.apply(init_with_xavier)
        self.additional_feat_extractor.apply(init_with_xavier)

    def feature_to_bbbox(self, loc_regress_layer, confidence_layer, input_feature):
        conf = confidence_layer(input_feature)
        loc = loc_regress_layer(input_feature)
        conf = conf.permute(0, 2, 3, 1).contiguous()
        num_batch = conf.shape[0]
        c_channels = int(conf.shape[1]*conf.shape[2]*conf.shape[3] / self.num_classes)
        conf = conf.view(num_batch, c_channels, self.num_classes)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(num_batch, c_channels, 4)
        return conf, loc

    def forward(self, input):
        confidence_list = []
        loc_list = []
        y = module_util.forward_from(self.base_net.base_net, 0, self.base_output_layer_indices[0]+1, input) #11 , 13
        confidence, loc = self.feature_to_bbbox(self.loc_regressor[0], self.classifier[0], y)
        confidence_list.append(confidence)
        loc_list.append(loc)
        y = module_util.forward_from(self.base_net.base_net, self.base_output_layer_indices[0]+1,
                                      self.base_output_layer_indices[1]+1, y)
        confidence, loc = self.feature_to_bbbox(self.loc_regressor[1], self.classifier[1], y)
        confidence_list.append(confidence)
        loc_list.append(loc)
        for i in range(len(self.additional_feat_extractor)):
            y = module_util.forward_from(self.additional_feat_extractor, i, i+1, y)
            confidence, loc = self.feature_to_bbbox(self.loc_regressor[i+2], self.classifier[i+2], y)
            confidence_list.append(confidence)
            loc_list.append(loc)
        confidences = torch.cat(confidence_list, 1)
        locations = torch.cat(loc_list, 1)
        # [Debug] check the output
        assert confidence.dim() == 3  # should be (N, num_priors, num_classes)
        assert locations.dim() == 3   # should be (N, num_priors, 4)
        assert confidences.shape[1] == locations.shape[1]
        assert locations.shape[2] == 4
        if not self.training:
            confidences = F.softmax(confidences, dim=2)
        return confidences, locations



