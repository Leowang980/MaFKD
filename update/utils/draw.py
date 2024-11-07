import numpy as np
import torch
from sklearn.manifold import TSNE
from alg.hetero import distribute
import matplotlib.pyplot as plt
from nets.resnets import ResNetCifar
import os

def draw_t_sne(args, dataloader):
    checkpoint_dir = '/home/ycli/FedOFLD/MyFed/checkpoint'
    checkpoints = [os.path.join(checkpoint_dir, f) 
                   for f in os.listdir(checkpoint_dir) if 'pth' in f]
    for checkpoint in checkpoints:
        check_point=torch.load(checkpoint)
        args.device=check_point['model']['conv1.weight'].device
        if 'HeteroFL' in checkpoint:
            print(checkpoint)
            selected_client=[0, 1, 2]
            local_param, _=distribute(args, selected_client, check_point['model'], args.model_level)
            all_pred=list()
            all_target=list()
            models=list()
            for i in selected_client:
                model=ResNetCifar(args, args.model_level[i]).to(args.device)
                model.load_state_dict(local_param[i])
                models.append(model)
            with torch.no_grad():
                for idx, (images, targets) in enumerate(dataloader):
                    images=images.to(args.device)
                    pred=list()
                    for model in models:
                        pred.append(model(images))
                    pred=sum(pred)/len(pred)
                    pred=pred.cpu().numpy()
                    targets=targets.cpu().numpy()
                    all_pred.append(pred)
                    all_target.extend(targets)
            
            all_pred=np.vstack(all_pred)
            tsne = TSNE(n_components=2, random_state=0)  
            tsne_results = tsne.fit_transform(all_pred)  
            
            # 可视化结果  
            plt.figure(figsize=(10, 10))  
            for i, c in enumerate(np.unique(all_target)):  
                plt.scatter(tsne_results[all_target == c, 0], tsne_results[all_target == c, 1], label=f'Class {c}', s=14)  
            plt.legend(loc = (-0.17,0.3), fontsize=16)  
            plt.axis('off')
            plt.savefig(checkpoint[:-8]+'.pdf',dpi=300, bbox_inches='tight')

        elif 'FedLFD_hetero' in checkpoint:
            continue
            print(checkpoint)
            all_pred=list()
            all_target=list()
            model=ResNetCifar(args, 1).to(args.device)
            model.load_state_dict(check_point['model'])
            with torch.no_grad():
                for idx, (images, targets) in enumerate(dataloader):
                    images=images.to(args.device)
                    pred=list()
                    pred=model.forward_feature(images)
                    b, c, h, w=pred.shape
                    pred=pred.view(b, c, h*w).mean(-1)
                    pred=pred.cpu().numpy()
                    targets=targets.cpu().numpy()
                    all_pred.append(pred)
                    all_target.extend(targets)
            
            all_pred=np.vstack(all_pred)
            tsne = TSNE(n_components=2, random_state=0)  
            tsne_results = tsne.fit_transform(all_pred)  
            
            # 可视化结果  
            plt.figure(figsize=(10, 10))  
            for i, c in enumerate(np.unique(all_target)):  
                plt.scatter(tsne_results[all_target == c, 0], tsne_results[all_target == c, 1], label=f'Class {c}', s=14)  
            #plt.legend(loc = (1.05,0.5), fontsize=16)  
            plt.tight_layout()
            plt.axis('off')
            plt.savefig(checkpoint[:-8]+'.pdf',dpi=300, bbox_inches='tight')
                    
                



        