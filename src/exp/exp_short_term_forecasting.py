from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas


warnings.filterwarnings('ignore')

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.token_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.token_len  # input_len = 2*token_len
            self.args.label_len = self.args.token_len
            if self.args.loss != 'MSE':
                self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                #print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_text) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)

                if self.args.loss == 'MSE':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print(f"\tModel outputs range: {outputs.min().item():.2f} to {outputs.max().item():.2f}")
                    print(f"\tTarget values range: {batch_y.min().item():.2f} to {batch_y.max().item():.2f}")
                    
                    # Check for NaN and stop training if detected
                    if torch.isnan(loss) or torch.isnan(outputs).any():
                        print("WARNING: NaN detected! Stopping training.")
                        return self.model
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.args.loss == 'MSE':
                vali_loss = self.vali(vali_data, vali_loader, criterion)
            else:
                vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        if self.args.loss == 'MSE':
            # Use data loader approach for MSE (like long-term forecasting)
            total_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_text) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)
                    
                    # Extract prediction part
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:, :]

                    loss = criterion(outputs, batch_y)
                    total_loss.append(loss.item())

            self.model.train()
            return np.average(total_loss)
        else:
            # Use original manual approach for other loss functions (exactly as it was)
            train_loader = vali_data  # vali_data is actually train_loader in this case
            x, _ = train_loader.dataset.last_insample_window()
            y = vali_loader.dataset.timeseries
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
            x = x.unsqueeze(-1)

            self.model.eval()
            with torch.no_grad():
                B, _, C = x.shape

                outputs = torch.zeros((B, self.args.seq_len, C)).float()  # .to(self.device)
                id_list = np.arange(0, B, 500)  # validation set size
                id_list = np.append(id_list, B)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        for i in range(len(id_list) - 1):
                            outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                            None,
                                                                            None).detach().cpu()
                else:
                    for i in range(len(id_list) - 1):
                            outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                            None,
                                                                            None).detach().cpu()
                pred = outputs[:, -self.args.token_len:, :]
                true = torch.from_numpy(np.array(y))
                batch_y_mark = torch.ones(true.shape)

                loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

            self.model.train()
            return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name

            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_trues = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompt_text) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, prompt_text)
                
                # Extract prediction part
                outputs = outputs[:, -self.args.token_len:, :]
                batch_y = batch_y[:, -self.args.token_len:, :]
                
                preds = outputs.detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()
                
                all_preds.append(preds)
                all_trues.append(trues)

            # Concatenate all predictions and truths
            preds = np.concatenate(all_preds, axis=0)
            trues = np.concatenate(all_trues, axis=0)

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.token_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return
