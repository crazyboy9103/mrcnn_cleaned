import torch
from torch.nn import Module
from torchvision.models.detection import MaskRCNN, faster_rcnn, mask_rcnn
from torch.utils.data import DataLoader
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision import models
import os

def collate_fn(batch):
    imgs, targets = [], []
    for img, target in batch:
        imgs.append(img)
        
        target['boxes'] = torch.unsqueeze(target['boxes'][0], 0)
        target['labels'] = torch.unsqueeze(target['labels'][0], 0)
        target['masks'] = torch.unsqueeze(target['masks'][0], 0)
        target['image_id'] = torch.unsqueeze(target['image_id'][0], 0)
        target['area'] = torch.unsqueeze(target['area'][0], 0)
        target['iscrowd'] = torch.unsqueeze(target['iscrowd'][0], 0)

        targets.append(target)
    return imgs, targets

class Model(Module):
    def __init__(self, num_classes, device, parallel, model_name, batch_size=8):
        super(Model, self).__init__()

        self.batch_size=batch_size
        self.num_classes=num_classes
        self.model = self.build_model(num_classes, device, parallel)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=0.005)
        
        self.device = device

        self.start_epoch = 0
        
        saved_models = [file for file in os.listdir() if file.endswith(".pt") and "mrcnn_model" in file]
        
        if model_name in os.listdir(): # if model name is found
            self.load(model_name)
            print(f"model loaded from {model_name}")
        
        else:
            saved_models = sorted(saved_models, 
                                key=lambda filename:int(filename.strip("mrcnn_model_").strip(".pt")))
            if saved_models:
                self.load(saved_models[-1])
                print(f"model loaded from {saved_models[-1]}")
                
    def build_model(self, num_classes, device, parallel):
        backbone = models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0), ))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 512
        model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        n_gpus = torch.cuda.device_count()
        if parallel:
            assert n_gpus >= 2
            model = torch.nn.DataParallel(model)
        model.to(device)
        return model

    def forward(self, images, targets):
        return self.model(images, targets)
    
    def fit(self, dataset, max_epochs):
        trainloader = DataLoader(dataset = dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)

        self.model.train()
        for e in range(self.start_epoch, self.start_epoch + max_epochs):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.001 * (0.95 ** e)
            
            
            for batch_idx, (images, targets) in enumerate(trainloader):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    str_buf = f"""
                    [EPOCH {e}/{self.start_epoch + max_epochs} {batch_idx}/{len(trainloader)} lr: {lr}] 
                    """
                    
                    for name, loss in loss_dict.items():
                        str_buf = str_buf + f" {name}: {loss:.6f}"
                    
                    print(str_buf)

            
            if e % 5 == 0:
                self.save(e)
        

     

    def save(self, epoch):
        path = f"mrcnn_model_{epoch}.pt"
        assert epoch != None
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save({
            'epoch':epoch,
            'model_state_dict':  model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print("Model saved at", path)

    def load(self, path):
        print("Loading model", path)
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        print("Starting from epoch", checkpoint['epoch'] + 1)

        if isinstance(self.model, torch.nn.DataParallel):
            try:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            except: 
                print("Failed to load from parallel ckpt")
                self.model = self.build_model(self.num_classes, self.device, parallel=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=0.005) 
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.005 * (0.95 ** checkpoint['epoch'])