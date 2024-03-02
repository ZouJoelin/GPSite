import numpy as np
import os, random, pickle
import datetime
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
import torch_geometric
from torch_geometric.loader import DataLoader
from data import *


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Metric(preds, labels):
    AUC = roc_auc_score(labels, preds)
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    AUPR = auc(recalls, precisions)
    return AUC, AUPR


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')


def train_and_predict(model_class, config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.run_id:
        run_id = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        run_id = args.run_id

    output_path = args.output_path + run_id + '/'
    os.makedirs(output_path, exist_ok = True)

    node_input_dim = config['node_input_dim']
    edge_input_dim = config['edge_input_dim']
    hidden_dim = config['hidden_dim']
    layer = config['layer']
    augment_eps = config['augment_eps']
    dropout = config['dropout']

    lr = config['lr']
    obj_max = config['obj_max']
    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    num_samples = config['num_samples']
    folds = config['folds']
    seed = config['seed']

    task = args.task

    if task == "BS":
        task_list = ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]
    else:
        task_list = [task]

    # Training
    if args.train:
        os.system(f'cp ./*.py {output_path}')
        os.system(f'cp ./*.sh {output_path}')

        log = open(output_path + 'train.log','w', buffering=1)
        Write_log(log, str(config) + '\n')

        CV_pred_dict = {} # 记录CV时每个蛋白的预测结果
        all_valid_metric = [] # 记录每一折的结果

        kf = KFold(n_splits = folds, shuffle=True, random_state=seed)

        with open(args.dataset_path + task + "_train.pkl", "rb") as f:  # ???
            train_data = pickle.load(f)
        for fold, (train_index, valid_index) in enumerate(kf.split(train_data)):
            Write_log(log, "\n========== Fold " + str(fold) + " ==========")

            train_dataset = ProteinGraphDataset(train_data, train_index, args, task_list)
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
            train_dataloader = DataLoader(train_dataset, batch_size = batch_size, sampler=sampler, shuffle=False, drop_last=True, num_workers=args.num_workers, prefetch_factor=2)

            valid_dataset = ProteinGraphDataset(train_data, valid_index, args, task_list)
            valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
            
            model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, task_list).to(device)

            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr, weight_decay=1e-5, eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs)  # ???

            loss_tr = nn.BCEWithLogitsLoss(reduction='none')

            if obj_max == 1:
                best_valid_metric = 0
            else:
                best_valid_metric = 1e9
            not_improve_epochs = 0

            for epoch in range(epochs):
                train_loss = 0
                train_num = 0
                model.train()

                train_pred = []
                train_y = []
                bar = tqdm(train_dataloader)
                for data in bar:
                    optimizer.zero_grad()
                    data = data.to(device)

                    outputs = model(data.X, data.node_feat, data.edge_index, data.seq, data.batch)

                    loss = loss_tr(outputs, data.y) * data.y_mask  # each data entry may only contain parts of task in args.task
                    loss = loss.sum() / data.y_mask.sum()  # ???
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    outputs = outputs.sigmoid() # [num_residue, num_task]

                    batch_train_pred = torch.masked_select(outputs, data.y_mask.bool()) # 所有任务混在一起
                    batch_train_y = torch.masked_select(data.y, data.y_mask.bool())

                    train_pred.append(batch_train_pred.detach().cpu().numpy())
                    train_y.append(batch_train_y.detach().cpu().numpy())

                    train_num += len(batch_train_y)
                    train_loss += len(batch_train_y) * loss.item()

                    bar.set_description('loss: %.4f' % (loss.item()))

                train_loss /= train_num
                train_pred = np.concatenate(train_pred)
                train_y = np.concatenate(train_y)
                train_metric = Metric(train_pred, train_y) # 一个epoch其实等于过了好几轮训练集
                torch.cuda.empty_cache()

                # Evaluate
                model.eval()
                valid_pred = [[] for task in task_list]
                valid_y = [[] for task in task_list]
                for data in tqdm(valid_dataloader):
                    data = data.to(device)
                    with torch.no_grad():
                        outputs = model(data.X, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid()

                    for i in range(len(task_list)):
                        batch_valid_y = torch.masked_select(data.y[:,i], data.y_mask[:,i].bool())
                        batch_valid_pred = torch.masked_select(outputs[:,i], data.y_mask[:,i].bool())
                        valid_y[i] += list(batch_valid_y.detach().cpu().numpy())
                        valid_pred[i] += list(batch_valid_pred.detach().cpu().numpy())

                valid_metrics = []
                for i in range(len(task_list)):
                    valid_metrics.append(Metric(valid_pred[i], valid_y[i]))
                valid_metrics = np.array(valid_metrics)
                valid_metric = valid_metrics.mean(0) # [average_AUC, average_AUPR]

                valid_auc = ",".join(list(valid_metrics[:,0].round(6).astype('str')))
                valid_aupr = ",".join(list(valid_metrics[:,1].round(6).astype('str')))

                if obj_max * (valid_metric[1]) > obj_max * best_valid_metric: # use AUPR
                    torch.save(model.state_dict(), output_path + 'fold%s.ckpt'%fold)
                    not_improve_epochs = 0
                    best_valid_metric = valid_metric[1]
                    
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_aupr: %.6f, valid_auc: %s, valid_aupr: %s'\
                    %(epoch,scheduler.get_last_lr()[0],train_loss,train_metric[0],train_metric[1],valid_auc,valid_aupr))
                else:
                    not_improve_epochs += 1
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_aupr: %.6f, valid_auc: %s, valid_aupr: %s, NIE +1 ---> %s'\
                    %(epoch,scheduler.get_last_lr()[0],train_loss,train_metric[0],train_metric[1],valid_auc,valid_aupr,not_improve_epochs))

                    if not_improve_epochs >= patience:
                        break

            # 用最好的epoch再测试一下validation，并存下一些预测结果
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, device)
            model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, task_list).to(device)
            model.load_state_dict(state_dict)
            model.eval()

            valid_pred = [[] for task in task_list]
            valid_y = [[] for task in task_list]
            for data in tqdm(valid_dataloader):
                data = data.to(device)
                with torch.no_grad():
                    outputs = model(data.X, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid()

                for i in range(len(task_list)):
                    batch_valid_y = torch.masked_select(data.y[:,i], data.y_mask[:,i].bool())
                    batch_valid_pred = torch.masked_select(outputs[:,i], data.y_mask[:,i].bool())
                    valid_y[i] += list(batch_valid_y.detach().cpu().numpy())
                    valid_pred[i] += list(batch_valid_pred.detach().cpu().numpy())

                # 导出CV的预测结果
                IDs = data.name
                outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
                for i, ID in enumerate(IDs):
                    CV_pred_dict[ID] = []
                    for j in range(len(task_list)):
                        CV_pred_dict[ID].append(list(outputs_split[i][:,j].detach().cpu().numpy()))

            valid_metrics = []
            for i in range(len(task_list)):
                valid_metrics.append(Metric(valid_pred[i], valid_y[i]))
            valid_metrics = np.array(valid_metrics)
            valid_metric = valid_metrics.mean(0) # [average_AUC, average_AUPR]

            valid_auc = ",".join(list(valid_metrics[:,0].round(6).astype('str')))
            valid_aupr = ",".join(list(valid_metrics[:,1].round(6).astype('str')))
                
            Write_log(log,'[fold %s] best_valid_auc_avg: %.6f, best_valid_aupr_avg: %.6f, best_valid_auc: %s, best_valid_aupr: %s'\
            %(fold, valid_metric[0], valid_metric[1], valid_auc, valid_aupr))

            all_valid_metric.append(valid_metrics[:,1]) # AUPR

        mean_valid_metric = np.mean(all_valid_metric, axis = 0) # 5折求平均
        Write_log(log,'CV mean metric: %s, mean metric over tasks: %.6f'%(",".join([str(round(x, 6)) for x in mean_valid_metric]), np.mean(mean_valid_metric)))

        with open(output_path + "CV_pred_dict.pkl", "wb") as f:
            pickle.dump(CV_pred_dict, f)

    # Test
    if args.test:
        if not args.train:
            log = open(output_path + 'test.log', 'w', buffering=1)
            Write_log(log,str(config)+'\n')

        with open(args.dataset_path + task + "_test.pkl", "rb") as f:
            test_data = pickle.load(f)
        test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args, task_list)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)

        models = []
        for fold in range(folds):
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, device)
            model = model_class(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, augment_eps, task_list).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        Write_log(log, 'model count: ' + str(len(models)))

        test_pred_dict = {} # 导出测试结果
        test_pred = [[] for task in task_list]
        test_y = [[] for task in task_list]
        for data in tqdm(test_dataloader):
            data = data.to(device)

            with torch.no_grad():
                outputs = [model(data.X, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid() for model in models]
                outputs = torch.stack(outputs,0).mean(0) # 5个模型预测结果求平均

            for i in range(len(task_list)):
                batch_test_y = torch.masked_select(data.y[:,i], data.y_mask[:,i].bool())
                batch_test_pred = torch.masked_select(outputs[:,i], data.y_mask[:,i].bool())
                test_y[i] += list(batch_test_y.detach().cpu().numpy())
                test_pred[i] += list(batch_test_pred.detach().cpu().numpy())

            # 导出预测结果
            IDs = data.name
            outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
            for i, ID in enumerate(IDs):
                test_pred_dict[ID] = []
                for j in range(len(task_list)):
                    test_pred_dict[ID].append(list(outputs_split[i][:,j].detach().cpu().numpy()))

        test_metrics = []
        for i in range(len(task_list)):
            test_metrics.append(Metric(test_pred[i], test_y[i]))
        test_metrics = np.array(test_metrics)
        test_metric = test_metrics.mean(0) # [average_AUC, average_AUPR]

        test_auc = ",".join(list(test_metrics[:,0].round(6).astype('str')))
        test_aupr = ",".join(list(test_metrics[:,1].round(6).astype('str')))
            
        Write_log(log,'test_auc_avg: %.6f, test_aupr_avg: %.6f, test_auc: %s, test_aupr: %s'\
        %(test_metric[0], test_metric[1], test_auc, test_aupr))

        with open(output_path + "test_pred_dict.pkl", "wb") as f:
            pickle.dump(test_pred_dict, f)

    log.close()        


