import torch, pickle, time, os
import numpy as np
import torch
from torch import nn

from param import args
from DataHander import DataHandler
from models.model import SDNet ,GCNModel 
from utils import load_model, save_model, fix_random_seed_as
from tqdm import tqdm
from models import gcn 
from models import diffusion_process as dp
from models.mlp_dim_reduction import MLPDimReduction
from Utils.Utils import *
import torch.nn.functional as F
import logging
import sys
import torch.distributed as dist
import math

print(torch.cuda.is_available())

class Coach:
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
        #self.device1 = torch.device('cuda' if args.cuda1 and torch.cuda.is_available() else 'cpu')
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user,self.n_item = self.handler.n_user, self.handler.n_item
        self.uiGraph = self.handler.ui_graph.to(self.device)
        self.uuGraph = self.handler.uu_graph.to(self.device)
        self.PreDiff_GCNModel = GCNModel(args,self.n_user, self.n_item).to(self.device)#实例化先扩散后GCN
        self.PostDiff_GCNModel = GCNModel(args,self.n_user, self.n_item).to(self.device)#实例化先GCN后扩散
        self.GCNModel= gcn.Dual_GCNModel(args,self.n_user, self.n_item).to(self.device)
        ### Build Diffusion process#
        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.PreSDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.PostSDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.PreDiffProcess=dp.DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)
        self.PostDiffProcess=dp.DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)
        self.DiffProcess=dp.DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)


        #self.MLPDimReduction1=MLPDimReduction(2*(self.args.n_hid),self.args.n_hid,self.device).to(self.device)#用MLP实现模型里两个用户嵌入向量合并降维
        #self.MLPDimReduction2=MLPDimReduction(2*(self.args.n_hid),self.args.n_hid,self.device).to(self.device)#用MLP实现模型里两个用户嵌入向量合并降维
        # self.MLPDimReduction3=MLPDimReduction(2*(self.args.n_hid),self.args.n_hid,self.device).to(self.device)#用MLP实现模型里两个用户嵌入向量合并降维
        # self.MLPDimReduction4=MLPDimReduction(2*(self.args.n_hid),self.args.n_hid,self.device).to(self.device)#用MLP实现模型里两个用户嵌入向量合并降维
       
       
        #定义优化器
        self.optimizer1 = torch.optim.Adam([
            {'params': self.PreDiff_GCNModel.parameters(),'weight_decay':0},
        ], lr=args.lr1)
        self.optimizer2 = torch.optim.Adam([
            {'params': self.PostDiff_GCNModel.parameters(),'weight_decay':0},
        ], lr=args.lr2)
        self.optimizer3 = torch.optim.Adam([
             {'params':  self.PreSDNet.parameters(), 'weight_decay': 0},
         ], lr=args.difflr1)
        self.optimizer4 = torch.optim.Adam([
             {'params':  self.PostSDNet.parameters(), 'weight_decay': 0},
         ], lr=args.difflr2)
        self.optimizer5 = torch.optim.Adam([
             {'params':  self.GCNModel.parameters(), 'weight_decay': 0},
         ], lr=args.lr)
        self.optimizer6 = torch.optim.Adam([
             {'params':  self.SDNet.parameters(), 'weight_decay': 0},
         ], lr=args.difflr)


         
        # self.optimizer5 = torch.optim.Adam([
        #      {'params':  self.MLPDimReduction1.parameters(), 'weight_decay': 0},
        #  ], lr=args.difflr)
        # self.optimizer6 = torch.optim.Adam([
        #      {'params':  self.MLPDimReduction2.parameters(), 'weight_decay': 0},
        #  ], lr=args.difflr)
        # self.optimizer7 = torch.optim.Adam([
        #      {'params':  self.MLPDimReduction3.parameters(), 'weight_decay': 0},
        #  ], lr=args.difflr)
        # self.optimizer8 = torch.optim.Adam([
        #      {'params':  self.MLPDimReduction4.parameters(), 'weight_decay': 0},
        #  ], lr=args.difflr)
        
        #定义调度器
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optimizer1,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optimizer2,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler3 = torch.optim.lr_scheduler.StepLR(
            self.optimizer3,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler4 = torch.optim.lr_scheduler.StepLR(
            self.optimizer4,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler5 = torch.optim.lr_scheduler.StepLR(
            self.optimizer5,
            step_size=args.decay_step,
            gamma=args.decay
        )
        self.scheduler6 = torch.optim.lr_scheduler.StepLR(
            self.optimizer6,
            step_size=args.decay_step,
            gamma=args.decay
        )
        # self.scheduler5 = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer5,
        #     step_size=args.decay_step,
        #     gamma=args.decay
        # )
        # self.scheduler6 = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer6,
        #     step_size=args.decay_step,
        #     gamma=args.decay
        # )
        # self.scheduler7 = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer7,
        #     step_size=args.decay_step,
        #     gamma=args.decay
        # )
        # self.scheduler8 = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer8,
        #     step_size=args.decay_step,
        #     gamma=args.decay
        # )
        self.train_loss = []
        self.his_recall = []
        self.his_ndcg  = []
    
    def train(self):
        args = self.args
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        best_recall1, best_ndcg1, best_epoch1, wait1 = 0, 0, 0, 0
        best_recall2, best_ndcg2, best_epoch2, wait2 = 0, 0, 0, 0
        start_time = time.time()
        for self.epoch in range(1, args.n_epoch + 1):
            epoch_losses = self.train_one_epoch()
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses
            ), flush=True)
            if self.epoch%5==0:
                recall, ndcg ,recall1, ndcg1 , recall2 , ndcg2 = self.test(self.testloader)
                
                #Record the history of recall and ndcg
                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                cur_best = recall + ndcg > best_recall + best_ndcg
                cur_best1 = recall1 + ndcg1 > best_recall1 + best_ndcg1
                cur_best2 = recall2 + ndcg2 > best_recall2 + best_ndcg2
                if cur_best:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                    wait = 0
                else:
                    wait += 1
                if cur_best1:
                    best_recall1, best_ndcg1, best_epoch = recall1, ndcg1, self.epoch
                    wait1 = 0
                else:
                    wait1 += 1
                if cur_best2:
                    best_recall2, best_ndcg2, best_epoch = recall2, ndcg2, self.epoch
                    wait2 = 0
                else:
                    wait2 += 1
                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, 10, recall, 10, ndcg))
                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, 20, recall1, 20, ndcg1))
                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, 40, recall2, 40, ndcg2))
                if args.model_dir and cur_best:
                    desc = args.save_name
                    perf = '' # f'N/R_{ndcg:.4f}/{hr:.4f}'
                    fname = f'{args.desc}_{desc}_{perf}.pth'
                    models_and_optimizers = {
                        'PreDiff_GCNModel': self.PreDiff_GCNModel,
                        'PostDiff_GCNModel': self.PostDiff_GCNModel,
                        'PreSDNet': self.PreSDNet,
                        'PostSDNet': self.PostSDNet,
                        'optimizer1': self.optimizer1,
                        'optimizer2': self.optimizer2,
                        'optimizer3': self.optimizer3,
                        'optimizer4': self.optimizer4
                    }
                    # save_model(models_and_optimizers, os.path.join(args.model_dir, fname))
            #if self.save_history:
                    #save_model(self.PreDiff_GCNModel, self.PostDiff_GCNModel,self.PreSDNet,self.PostSDNet, os.path.join(args.model_dir, fname), self.optimizer1,self.optimizer2,self.optimizer3,self.optimizer4)
            # if self.save_history:
            #     self.saveHistory()


            # if wait >= args.patience:
            #     print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
            #     break

        print(f'Best  Recall@{10} {best_recall:.6f}, NDCG@{10} {best_ndcg:.6f},', flush=True)
        print(f'Best  Recall@{20} {best_recall1:.6f}, NDCG@{20} {best_ndcg1:.6f},', flush=True)
        print(f'Best  Recall@{40} {best_recall2:.6f}, NDCG@{40} {best_ndcg2:.6f},', flush=True)
    def train_one_epoch(self):
        self.PreDiff_GCNModel.train()
        self.PostDiff_GCNModel.train()
        self.PreSDNet.train()
        self.PostSDNet.train()
        self.GCNModel.train()
        self.SDNet.train()
        #self.MLPDimReduction1.train()
        #self.MLPDimReduction2.train()
        # self.MLPDimReduction3.train()
        # self.MLPDimReduction4.train()
        dataloader = self.train_loader
        epoch_losses = [0] * 3
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()
       
        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()
            pre_user_embeddings=self.PreDiff_GCNModel.user_embeddings 
            pre_item_embeddings=self.PreDiff_GCNModel.item_embeddings 
            # self.PreDiffProcess = self.PreDiffProcess.cpu()
            # self.PreSDNet = self.PreSDNet.cpu() 
            # pre_user_embeddings = pre_user_embeddings.cpu()
            prediff_user_embeddings,prediff_weight=self.PreDiffProcess.get_output(self.PreSDNet, pre_user_embeddings, args.reweight)
            #prediff_user_embeddings=prediff_uuterms["pred_xstart"] 
            
            #prediff_loss=prediff_uuterms["loss"].mean().to(self.device)
            # tmp = pre_user_embeddings
            # tmp[user_idx] = prediff_user_embeddings
            pre_uiEmbeds,pre_uuEmbeds= self.PreDiff_GCNModel(self.uiGraph,self.uuGraph,prediff_user_embeddings,pre_item_embeddings,True)#先扩散GCN聚合扩散后的用户嵌入
            pre_uEmbeds = pre_uiEmbeds[:self.n_user]
            pre_iEmbeds = pre_uiEmbeds[self.n_user:]
            mse1 = mean_flat((pre_user_embeddings - pre_uuEmbeds) ** 2)
            prediff_loss = (prediff_weight * mse1).mean()
            prediff_user = pre_uEmbeds[user_idx]
            user1=prediff_user+pre_uuEmbeds[user_idx]
            prediff_pos = pre_iEmbeds[pos_idx]
            prediff_neg = pre_iEmbeds[neg_idx]
            # input_users1=torch.concat([prediff_user,pre_uuEmbeds[user_idx]],axis=1)
            # user1=self.MLPDimReduction1(input_users1)
            post_user_embeddings=self.PostDiff_GCNModel.user_embeddings 
            post_item_embeddings=self.PostDiff_GCNModel.item_embeddings 
            post_uiEmbeds,post_uuEmbeds= self.PostDiff_GCNModel(self.uiGraph,self.uuGraph,post_user_embeddings,post_item_embeddings,True)#先GCN聚合扩散后的用户嵌入
            postdiff_user_embeddings,postdiff_weight=self.PostDiffProcess.get_output(self.PostSDNet, post_uuEmbeds, args.reweight)#得到扩散损失权重
            #postdiff_user_embeddings=postdiff_uuterms["pred_xstart"] 
            #postdiff_loss=postdiff_uuterms["loss"].mean()
            mse2 = mean_flat((post_user_embeddings - postdiff_user_embeddings) ** 2)
            postdiff_loss=(postdiff_weight * mse2).mean() 
            post_uEmbeds = post_uiEmbeds[:self.n_user]
            post_iEmbeds = post_uiEmbeds[self.n_user:]
            postdiff_user = post_uEmbeds[user_idx]
            postdiff_uu_userembeddings=postdiff_user_embeddings[user_idx]
            user2=postdiff_user+postdiff_uu_userembeddings
            postdiff_pos = post_iEmbeds[pos_idx]
            postdiff_neg = post_iEmbeds[neg_idx]
            # input_users2=torch.concat([postdiff_user,postdiff_user_embeddings],axis=1)
            # user2=self.MLPDimReduction2(input_users2)
            contrastive_loss=self.calc_contrastiveloss(user1,user2)
            user_embeddings=pre_uuEmbeds+pre_uEmbeds+postdiff_user_embeddings+post_uEmbeds
            item_embeddings= pre_iEmbeds+post_iEmbeds
            uiEmbeds,uuEmbeds= self.GCNModel(self.uiGraph,self.uuGraph,user_embeddings,item_embeddings,True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]
            uu_user=uuEmbeds[user_idx]
            diff_user_embeddings,diff_weight=self.DiffProcess.get_output(self.SDNet, uu_user, args.reweight)
            mse = mean_flat((uu_user - diff_user_embeddings) ** 2)
            diff_loss=(diff_weight * mse).mean() 

            ui_user=uEmbeds[user_idx]
            user=diff_user_embeddings+ui_user
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]
         
            # input_users=torch.concat([user1,user2],axis=1)
            # user=self.MLPDimReduction1(input_users)
            # # input_items=torch.concat([pre_iEmbeds,post_iEmbeds],axis=1)
            # # iEmbeds=self.MLPDimReduction2(input_items)
            # iEmbeds=(pre_iEmbeds+post_iEmbeds)/2
            # pos=iEmbeds[pos_idx]
            # neg=iEmbeds[neg_idx]
        
            # scoreDiff=pairPredict(user, pos, neg)
            # bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch_size
            # regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * args.reg)/args.batch_size
            # loss=bprLoss+regLoss
            # losses = [bprLoss.item(), regLoss.item()]

            # diffloss=prediff_loss+postdiff_loss
            # loss = loss+diffloss+contrastive_loss
            # losses.append(diffloss.item())
            # user=(user1+user2)/2
            # pos=(prediff_pos+prediff_pos)/2
            # neg=(prediff_neg+postdiff_neg)/2
            scoreDiff1 = pairPredict(user1, prediff_pos, prediff_neg)
            bprLoss1 = - (scoreDiff1).sigmoid().log().sum() / args.batch_size
            regLoss1 = ((torch.norm(user1) ** 2 + torch.norm(prediff_pos) ** 2 + torch.norm(prediff_neg) ** 2) * args.reg1)/args.batch_size
            scoreDiff2 = pairPredict(user2, postdiff_pos, postdiff_neg)
            bprLoss2 = - (scoreDiff2).sigmoid().log().sum() / args.batch_size
            regLoss2 = ((torch.norm(user2) ** 2 + torch.norm(postdiff_pos) ** 2 + torch.norm(postdiff_neg) ** 2) * args.reg2)/args.batch_size
            bprLoss=bprLoss1 + bprLoss2
            regLoss=regLoss1 + regLoss2
            scoreDiff = pairPredict(user, pos, neg)
            last_bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch_size
            last_regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * args.reg)/args.batch_size

            loss = bprLoss + regLoss+last_bprLoss+last_regLoss
            losses = [bprLoss.item(), regLoss.item()]
            diffloss=prediff_loss+postdiff_loss+diff_loss
            loss = loss+diffloss+contrastive_loss
            
            losses.append(diffloss.item())
           
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()
            self.optimizer4.zero_grad()
            self.optimizer5.zero_grad()
            self.optimizer6.zero_grad()

            #self.optimizer5.zero_grad()
            #self.optimizer6.zero_grad()
            # self.optimizer7.zero_grad()
            # self.optimizer8.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            self.optimizer3.step()
            self.optimizer4.step()
            self.optimizer5.step()
            self.optimizer6.step()

            #self.optimizer5.step()
            #self.optimizer6.step()
            # self.optimizer7.step()
            # self.optimizer8.step()
            epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]
        if self.scheduler1 is not None:
            self.scheduler1.step()
            self.scheduler2.step()
            self.scheduler3.step()
            self.scheduler4.step()
            self.scheduler5.step()
            self.scheduler6.step()

            #self.scheduler5.step()
            # self.scheduler6.step()
            # self.scheduler7.step()
            # self.scheduler8.step()
        epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(
            time_elapsed ))
        return epoch_losses



    def calc_contrastiveloss(self,user,gcn_user):
        cos_sim = F.cosine_similarity(user.unsqueeze(1), gcn_user.unsqueeze(0), dim=-1)/self.args.tau
        numerator = torch.exp(torch.diag(cos_sim))  
        denominator = torch.sum(torch.exp(cos_sim), dim=1)  
        loss = -torch.log(numerator / denominator)
        
        return torch.mean(loss)

    def calcRes(self, topLocs, tstLocs, batIds): 
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg


    def test(self,dataloader):
        self.PreDiff_GCNModel.eval()
        self.PostDiff_GCNModel.eval()
        self.PreSDNet.eval()
        self.PostSDNet.eval()
        self.GCNModel.eval()
        self.SDNet.eval()

        #self.MLPDimReduction1.eval()
        #self.MLPDimReduction2.eval()
        # self.MLPDimReduction3.eval()
        # self.MLPDimReduction4.eval()
        Recall, NDCG = [0] * 2
        Recall1, NDCG1 = [0] * 2
        Recall2, NDCG2 = [0] * 2
        num = dataloader.dataset.__len__()
       
        since = time.time()
        with torch.no_grad():
            pre_user_embeddings=self.PreDiff_GCNModel.user_embeddings 
            pre_item_embeddings=self.PreDiff_GCNModel.item_embeddings 
            prediff_upredict=self.PreDiffProcess.p_sample(self.PreSDNet, pre_user_embeddings, args.sampling_steps, args.sampling_noise)
            pre_uiEmbeds,pre_uuEmbeds= self.PreDiff_GCNModel(self.uiGraph,self.uuGraph,prediff_upredict,pre_item_embeddings,True)
            pre_uEmbeds = pre_uiEmbeds[:self.n_user]
            pre_iEmbeds = pre_uiEmbeds[self.n_user:]
            post_user_embeddings=self.PostDiff_GCNModel.user_embeddings 
            post_item_embeddings=self.PostDiff_GCNModel.item_embeddings 
            post_uiEmbeds,post_uuEmbeds= self.PostDiff_GCNModel(self.uiGraph,self.uuGraph,post_user_embeddings,post_item_embeddings,True)
            postdiff_upredict = self.PostDiffProcess.p_sample(self.PostSDNet, post_uuEmbeds, args.sampling_steps, args.sampling_noise)
            post_uEmbeds = post_uiEmbeds[:self.n_user]
            post_iEmbeds = post_uiEmbeds[self.n_user:]
            user_embeddings=pre_uuEmbeds+pre_uEmbeds+postdiff_upredict+post_uEmbeds
            item_embeddings=pre_iEmbeds+post_iEmbeds
            uiEmbeds,uuEmbeds=self.GCNModel(self.uiGraph,self.uuGraph,user_embeddings,item_embeddings,True)
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long().cuda()
                trnMask = trnMask.cuda()
                # pre_uEmbeds = pre_uiEmbeds[:self.n_user]
                # pre_iEmbeds = pre_uiEmbeds[self.n_user:]
                # pre_uuemb = pre_uEmbeds[user_idx]
                # pre_user = pre_uuEmbeds[user_idx]
                # user1=pre_uuemb+pre_user
                # input_users1=torch.concat([pre_uuemb,pre_uuEmbeds[user_idx]],axis=1)
                # user1=self.MLPDimReduction1(input_users1)
                # post_uEmbeds = post_uiEmbeds[:self.n_user]
                # post_iEmbeds = post_uiEmbeds[self.n_user:]
                # post_uuemb = post_uEmbeds[user_idx]
                # #post_user = post_uuEmbeds[user_idx]
                # post_user=postdiff_upredict[user_idx]
                # user2=post_uuemb+post_user
                # input_users2=torch.concat([pre_uuemb,postdiff_upredict],axis=1)
                # user2=self.MLPDimReduction2(input_users2)
                # input_users=torch.concat([user1,user2],axis=1)
                # user=self.MLPDimReduction1(input_users)
                # iEmbeds=(pre_iEmbeds+post_iEmbeds)/2
                # input_items=torch.concat([pre_iEmbeds,post_iEmbeds],axis=1)
                # iEmbeds=self.MLPDimReduction2(input_items)
                UI_uEmbeds = uiEmbeds[:self.n_user]
                UI_iEmbeds = uiEmbeds[self.n_user:]
                # pre_iEmbeds = pre_uiEmbeds[self.n_user:]
                # pre_uuemb = pre_uEmbeds[user_idx]
                # pre_user = pre_uuEmbeds[user_idx]#
                #user=user1+user2
                #iEmbeds=pre_iEmbeds+post_iEmbeds
                UU_uEmbeds=uuEmbeds[user_idx]
                upredict = self.DiffProcess.p_sample(self.SDNet, UU_uEmbeds, args.sampling_steps, args.sampling_noise)#先聚合再扩散的UU图用户嵌入
                user=upredict+UI_uEmbeds[user_idx]
                
                allPreds = t.mm(user, t.transpose(UI_iEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                _, topLocs = t.topk(allPreds, 10)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall+= recall
                NDCG+=ndcg

                _, topLocs1 = t.topk(allPreds, 20)
                recall1, ndcg1 = self.calcRes(topLocs1.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall1+= recall1
                NDCG1+=ndcg1

                _, topLocs2 = t.topk(allPreds, 40)
                recall2, ndcg2 = self.calcRes(topLocs2.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall2+= recall2
                NDCG2+=ndcg2
            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(
            time_elapsed ))
            Recall = Recall/num
            NDCG = NDCG/num
            Recall1 = Recall1/num
            NDCG1 = NDCG1/num
            Recall2 = Recall2/num
            NDCG2 = NDCG2/num
        return Recall, NDCG ,Recall1, NDCG1,Recall2, NDCG2

 

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['Recall'] = self.his_recall
        history['NDCG'] = self.his_ndcg
        ModelName = "SDR"
        desc = args.save_name
        perf = ''  # f'N/R_{ndcg:.4f}/{hr:.4f}'
        fname = f'{args.desc}_{desc}_{perf}.his'

        with open('./History/' + args.dataset + '/' + fname, 'wb') as fs:
            pickle.dump(history, fs)

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


if __name__ == "__main__":
    #
    #dist.init_process_group(backend='nccl')
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(local_rank)
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    fix_random_seed_as(args.seed)

    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(handler)
    # train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
