import torch
import os
import time
import copy
from lib.utils import get_logger
from lib.metrics import All_Metrics
from lib.utils import print_model_parameters


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False :
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info(args)


    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(val_dataloader):
                data = data
                label = target[..., :self.args.output_dim]
                output = self.model(data)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    label = self.scaler.inverse_transform(label)              
                loss = self.loss(output.cuda(), label)              
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for _, (data, target) in enumerate(self.train_loader):
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            data = data
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
                label = self.scaler.inverse_transform(label)    
            loss = self.loss(output.cuda(), label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss


    def train(self):
        try:
            best_model = None
            not_improved_count = 0
            best_loss = float('inf')
            print_model_parameters(self.model, self.logger, only_num=False)
            self.logger.info("Start training")
            for epoch in range(0, self.args.epochs + 1):
                epoch_time = time.time()
                train_epoch_loss = self.train_epoch(epoch)
                print("***************Training Time: {:.4f} secs/epoch***************".format(time.time()-epoch_time))
                if self.val_loader == None:
                    val_dataloader = self.test_loader
                else:
                    val_dataloader = self.val_loader
                val_epoch_loss = self.val_epoch(epoch, val_dataloader)
                if train_epoch_loss > 1e6:
                    self.logger.warning('Gradient explosion detected. Ending...')
                    break
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False
                # early stop
                if self.args.early_stop:
                    if not_improved_count == self.args.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.args.early_stop_patience))
                        break
                # save the best state
                if best_state == True:
                    self.logger.info('*********************************Current best model saved!')
                    best_model = copy.deepcopy(self.model.state_dict())
            #save the best model to file
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        
        
        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        epoch_time = time.time()
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)
        print("***************Inference Time: {:.4f} secs/epoch***************".format(time.time()-epoch_time))

     
    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data = data
                label = target[..., :args.output_dim]
                output = model(data)
                y_true.append(label)
                y_pred.append(output)
        if args.real_value:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        else:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
        #np.save('./{}_{}_true.npy'.format(args.model, args.dataset), y_true.cpu().numpy())
        #np.save('./{}_{}_pred.npy'.format(args.model, args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.6f}, RMSE: {:.6f}, MAPE: {:.6f}%".format(
                    mae, rmse, mape*100))
