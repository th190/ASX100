import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from utils import get_model_inputs

class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size, rnn_num_layers, rnn_type):
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                num_layers = rnn_num_layers,
                input_size=input_feature_len,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                num_layers = rnn_num_layers,
                input_size=input_feature_len,
                hidden_size=hidden_size,
                batch_first=True,
            )
        
    def forward(self, input_seq, hidden):
        rnn_out, hidden = self.rnn(input_seq, hidden)
        return rnn_out, hidden
    

# class Seq2SeqAttnDecoder(nn.Module):
#     def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type):
#         super(Seq2SeqAttnDecoder, self).__init__()
#         self.rnn_type = rnn_type
#         if rnn_type == 'gru':
#             self.rnn = nn.GRU(
#                 num_layers = rnn_num_layers,
#                 input_size= hidden_size+1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
#         else:
#             self.rnn = nn.LSTM(
#                 num_layers = rnn_num_layers,
#                 input_size=hidden_size+1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
        
#         if attn_dim is None:
#             attn_dim = hidden_size
#         self.attn_w1 = nn.Linear(hidden_size, attn_dim)
#         self.attn_w2 = nn.Linear(hidden_size, attn_dim)
#         self.attn_out = nn.Linear(attn_dim, 1)
        
#         self.out = nn.Linear(hidden_size, 1)
        
#     def forward(self, input_seq, prev_hidden, encoder_outputs, return_attn=False):
#         if self.rnn_type == 'gru':
#             attn_score = self.attn_out(torch.tanh(self.attn_w1(prev_hidden.permute(1,0,2)) + self.attn_w2(encoder_outputs)))
#         else:
#             attn_score = self.attn_out(torch.tanh(self.attn_w1(prev_hidden[0].permute(1,0,2)) + self.attn_w2(encoder_outputs)))
            
#         attn_weight = F.softmax(attn_score, dim=1)
#         context = torch.sum(attn_weight*encoder_outputs, axis=1)
        
#         output, hidden = self.rnn(torch.cat((input_seq, context.unsqueeze(1)), -1), prev_hidden)        

#         output = self.out(output)
#         if return_attn:
#             return output, hidden, attn_weight
#         return output, hidden


# tanh attention
class Seq2SeqAttnDecoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type):
        super(Seq2SeqAttnDecoder, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                num_layers = rnn_num_layers,
                input_size= 1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                num_layers = rnn_num_layers,
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        
        if attn_dim is None:
            attn_dim = hidden_size
        self.attn_w1 = nn.Linear(hidden_size, attn_dim)
        self.attn_w2 = nn.Linear(hidden_size, attn_dim)
        self.attn_out = nn.Linear(attn_dim, 1)
        
        self.out = nn.Linear(hidden_size*2, 1)
        
    def forward(self, input_seq, prev_hidden, encoder_outputs, return_attn=False):
#         print(encoder_outputs.size())
        if self.rnn_type == 'gru':
            attn_score = self.attn_out(torch.tanh(self.attn_w1(prev_hidden.permute(1,0,2)) + self.attn_w2(encoder_outputs)))
        else:
            attn_score = self.attn_out(torch.tanh(self.attn_w1(prev_hidden[0].permute(1,0,2)) + self.attn_w2(encoder_outputs)))
            
        attn_weight = F.softmax(attn_score, dim=1)
#         print(attn_weight.size())
        context = torch.sum(attn_weight*encoder_outputs, axis=1)
        
        output, hidden = self.rnn(input_seq, prev_hidden)        
        output = torch.cat((output, context.unsqueeze(1)), -1)

        output = self.out(output)
        if return_attn:
            return output, hidden, attn_weight
        return output, hidden

# #Pytorch MultiheadAttention
# class Seq2SeqAttnDecoder(nn.Module):
#     def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type):
#         super(Seq2SeqAttnDecoder, self).__init__()
#         self.rnn_type = rnn_type
#         if rnn_type == 'gru':
#             self.rnn = nn.GRU(
#                 num_layers = rnn_num_layers,
#                 input_size= 1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
#         else:
#             self.rnn = nn.LSTM(
#                 num_layers = rnn_num_layers,
#                 input_size=1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
        

#         self.attention = nn.MultiheadAttention(hidden_size, 1)
#         self.out = nn.Linear(hidden_size*2, 1)
        
#     def forward(self, input_seq, prev_hidden, encoder_outputs, return_attn=False):
#         if self.rnn_type == 'gru':
#             context, attn_weight = self.attention(prev_hidden.permute(1,0,2), encoder_outputs, encoder_outputs)
#         else:
#             context, attn_weight = self.attention(prev_hidden[0].permute(1,0,2), encoder_outputs, encoder_outputs)        
        
#         output, hidden = self.rnn(input_seq, prev_hidden)        
#         output = torch.cat((output, context), -1)

#         output = self.out(output)
#         if return_attn:
#             return output, hidden, attn_weight
#         return output, hidden

# #Dot product attention
# class Seq2SeqAttnDecoder(nn.Module):
#     def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type):
#         super(Seq2SeqAttnDecoder, self).__init__()
#         self.rnn_type = rnn_type
#         if rnn_type == 'gru':
#             self.rnn = nn.GRU(
#                 num_layers = rnn_num_layers,
#                 input_size= 1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
#         else:
#             self.rnn = nn.LSTM(
#                 num_layers = rnn_num_layers,
#                 input_size=1,
#                 hidden_size=hidden_size,
#                 batch_first=True,
#             )
        
#         if attn_dim is None:
#             attn_dim = hidden_size
#         self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.out = nn.Linear(hidden_size*2, 1)
        
#     def forward(self, input_seq, prev_hidden, encoder_outputs, return_attn=False):
#         if self.rnn_type == 'gru':
#             attn_weight = F.softmax(torch.bmm(self.w_k(encoder_outputs), self.w_q(prev_hidden).permute(1,2,0)), dim=1).permute(0,2,1)
#         else:
#             attn_weight = F.softmax(torch.bmm(self.w_k(encoder_outputs), self.w_q(prev_hidden[0]).permute(1,2,0)), dim=1).permute(0,2,1)
            
#         context = torch.bmm(attn_weight, self.w_v(encoder_outputs))

#         output, hidden = self.rnn(input_seq, prev_hidden)        
#         output = torch.cat((output, context), -1)

#         output = self.out(output)
#         if return_attn:
#             return output, hidden, attn_weight
#         return output, hidden

class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim=None, rnn_type='gru'):
        super(Seq2SeqDecoder, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                num_layers = rnn_num_layers,
                input_size= 1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                num_layers = rnn_num_layers,
                input_size=1,
                hidden_size=hidden_size,
                batch_first=True,
            )
        
        self.out = nn.Linear(hidden_size, 1)
        
    def forward(self, input_seq, prev_hidden, encoder_outputs, return_attn=False):
        output, hidden = self.rnn(input_seq, prev_hidden)        

        output = self.out(output)
        if return_attn:
            return output, hidden, None
        return output, hidden


    
class Seq2Seq:
    def __init__(self, input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim=None, rnn_type='gru', use_attn=False, device='cuda'):
        self.input_feature_len = input_feature_len
        self.hidden_size = hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.attn_dim = attn_dim
        self.rnn_type = rnn_type
        self.device = device
        self.encoder = Seq2SeqEncoder(input_feature_len, hidden_size, rnn_num_layers, rnn_type).to(device)
        if use_attn:
            self.decoder = Seq2SeqAttnDecoder(input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type).to(device)
        else:
            self.decoder = Seq2SeqDecoder(input_feature_len, hidden_size, rnn_num_layers, seq_length, attn_dim, rnn_type).to(device)
        self.criterion = nn.MSELoss() 
    
    def train_step(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio):
        batch_size = input_tensor.size(0)
        if self.rnn_type == 'gru':
            encoder_hidden = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device)
        else:
            encoder_hidden = (torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device),
                             torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device))
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)

        encoder_outputs = torch.zeros(batch_size, input_length, self.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[:,ei,:].unsqueeze(1), encoder_hidden)
            encoder_outputs[:,ei,:] = encoder_output[:,0,:]

        decoder_input = input_tensor[:,-1,3].unsqueeze(1).unsqueeze(-1)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output[:,0,:], target_tensor[:,di,:])
                decoder_input = target_tensor[:,di,:].unsqueeze(1)  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.detach()  # detach from history as input
                loss += self.criterion(decoder_output[:,0,:], target_tensor[:,di,:])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def predict(self, input_tensor, output_length, return_attn=False):
        with torch.no_grad():
            batch_size = input_tensor.size(0)
            input_length = input_tensor.size(1)
            if self.rnn_type == 'gru':
                encoder_hidden = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device)
            else:
                encoder_hidden = (torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device),
                                 torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size, device=self.device))
                
            encoder_outputs = torch.zeros(batch_size, input_length, self.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[:,ei,:].unsqueeze(1), encoder_hidden)
                encoder_outputs[:,ei,:] = encoder_output[:,0,:]

            decoder_input = input_tensor[:,-1,3].unsqueeze(1).unsqueeze(-1)
            decoder_hidden = encoder_hidden
            predictions = torch.zeros(batch_size, output_length, 1, device=self.device)
            
            attn_weights = []
            for di in range(output_length):
                if return_attn:
                    decoder_output, decoder_hidden, attn_weight = self.decoder(decoder_input, decoder_hidden, encoder_outputs, return_attn)
                    attn_weights.append(attn_weight)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.detach()
                predictions[:,di,:] = decoder_output[:,0,:]
        if return_attn:
            return predictions, attn_weights#, encoder_outputs, decoder_hidden        
        return predictions
    
    def validate(self, input_tensor, y_val, output_length):
        with torch.no_grad():
            y_val_pred = self.predict(input_tensor, output_length)
            loss_val = self.criterion(y_val_pred, y_val)
        return loss_val

    def save_model(self, path):
        torch.save({'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict()}, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder.eval()
        self.decoder.eval()

    def test_predict(self, stock, test_data, scalers, n_back, n_forward):
        predictions = []
        test_data_stock = np.array(test_data[stock])
        true_values = test_data_stock[n_back:,3]
        test_input = torch.from_numpy(test_data_stock.astype('float32')).to(self.device)
        
        idx = 0
        while len(predictions) < len(true_values):
            input_tensor = test_input[idx:idx+n_back].unsqueeze(0)
            next_pred = self.predict(input_tensor, output_length=n_forward)
            predictions.extend(next_pred[0,:,0].tolist())
            idx += n_forward
        
        predictions = np.array(predictions[:len(true_values)])*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        true_values = true_values*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        return predictions, true_values
    
    def get_metrics(self, stock_list, test_data, scalers, n_back, n_forward, day_ahead=None):
        model_metrics = pd.DataFrame()

        for stock in stock_list:
            x_test, y_test = get_model_inputs(stock, test_data, n_back, n_forward)
            y_test = y_test*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            y_test_pred = self.predict(torch.from_numpy(x_test.astype(np.float32)).to(self.device), n_forward).cpu().numpy()[:,:,0]*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            
            rmse = np.sqrt(np.mean((y_test - y_test_pred)**2, axis=0))
            mae = np.mean(np.abs(y_test - y_test_pred), axis=0)
            mape = np.mean(np.abs(y_test - y_test_pred)/y_test, axis=0)
            
            if day_ahead is not None:
                rmse = rmse[day_ahead]
                mae = mae[day_ahead]
                mape = mape[day_ahead]
            else:
                rmse = np.mean(rmse)
                mae = np.mean(mae)
                mape = np.mean(mape)

            metrics = pd.DataFrame({'stock': [stock], 'rmse': [rmse], 'mae': [mae], 'mape': [mape]})
            model_metrics = pd.concat([model_metrics, metrics], axis=0)

        model_metrics = model_metrics.reset_index(drop=True)  
        return model_metrics

#     def get_metrics(self, stock_list, test_data, scalers, n_back, n_forward):
#         model_metrics = pd.DataFrame()

#         for stock in stock_list:
#             predictions, true_values = self.test_predict(stock, test_data, scalers, n_back, n_forward)

#             rmse = mean_squared_error(true_values, predictions, squared=False)
#             mae = mean_absolute_error(true_values, predictions)
#             mape = np.mean(np.abs((true_values - predictions)/true_values))

#             metrics = pd.DataFrame({'stock': [stock], 'rmse': [rmse], 'mae': [mae], 'mape': [mape]})
#             model_metrics = pd.concat([model_metrics, metrics], axis=0)

#         model_metrics = model_metrics.reset_index(drop=True)  
#         return model_metrics
    
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:,-1,:]) 
        return out
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # (batch_dim, seq_dim, feature_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hn = self.gru(x)
        out = self.fc(out[:,-1,:]) 
        return out
    
    
class RNNmodel:
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, rnn_type='gru', device='cuda'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device            
        self.rnn = GRU(input_dim, hidden_dim, num_layers, output_dim).to(device)
        if rnn_type == 'lstm':
            self.rnn = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
        self.criterion = nn.MSELoss() 
    
    def train_step(self, input_tensor, target_tensor, optimizer):
        optimizer.zero_grad()
        y_train_pred = self.rnn(input_tensor)
        loss = self.criterion(y_train_pred, target_tensor)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, input_tensor):
        with torch.no_grad():
            predictions = self.rnn(input_tensor)
        return predictions
    
    def validate(self, input_tensor, y_val):
        with torch.no_grad():
            y_val_pred = self.predict(input_tensor)
            loss_val = self.criterion(y_val_pred, y_val)
        return loss_val

    def save_model(self, path):
        torch.save(self.rnn.state_dict(), path)
    
    def load_model(self, path):
        self.rnn.load_state_dict(torch.load(path))
        self.rnn.eval()

    def test_predict(self, stock, test_data, scalers, n_back, n_forward):
        predictions = []
        test_data_stock = np.array(test_data[stock])
        true_values = test_data_stock[n_back:,3]
        test_input = torch.from_numpy(test_data_stock.astype('float32')).to(self.device)
        
        idx = 0
        while len(predictions) < len(true_values):
            input_tensor = test_input[idx:idx+n_back].unsqueeze(0)
            next_pred = self.predict(input_tensor)
            predictions.extend(next_pred[0].tolist())
            idx += n_forward

        predictions = np.array(predictions[:len(true_values)])*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        true_values = true_values*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        return predictions, true_values
    
    def get_metrics(self, stock_list, test_data, scalers, n_back, n_forward, day_ahead=None):
        model_metrics = pd.DataFrame()

        for stock in stock_list:
            x_test, y_test = get_model_inputs(stock, test_data, n_back, n_forward)
            y_test = y_test*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            y_test_pred = self.predict(torch.from_numpy(x_test.astype(np.float32)).to(self.device)).cpu().numpy()*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            
            rmse = np.sqrt(np.mean((y_test - y_test_pred)**2, axis=0))
            mae = np.mean(np.abs(y_test - y_test_pred), axis=0)
            mape = np.mean(np.abs(y_test - y_test_pred)/y_test, axis=0)

            if day_ahead is not None:
                rmse = rmse[day_ahead]
                mae = mae[day_ahead]
                mape = mape[day_ahead]
            else:
                rmse = np.mean(rmse)
                mae = np.mean(mae)
                mape = np.mean(mape)

            metrics = pd.DataFrame({'stock': [stock], 'rmse': [rmse], 'mae': [mae], 'mape': [mape]})
            model_metrics = pd.concat([model_metrics, metrics], axis=0)

        model_metrics = model_metrics.reset_index(drop=True)  
        return model_metrics



class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, max_iter=10000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = MLPRegressor(hidden_layer_sizes=(hidden_dim,), solver='sgd', max_iter=max_iter, alpha=0)

        
    def predict(self, input):
        return self.model.predict(input)
    
    def save_model(self, path):
        dump(self.model, path)
    
    def load_model(self, path):
        self.model = load(path)
        
    def fit(self, x, y):
        self.model.fit(x,y)
        
    def test_predict(self, stock, test_data, scalers, n_back, n_forward):
        predictions = []
        test_data_stock = np.array(test_data[stock])
        true_values = test_data_stock[n_back:,3]
        test_input = test_data_stock

        idx = 0
        while len(predictions) < len(true_values):
            input_tensor = test_input[idx:idx+n_back].reshape(1,-1)
            next_pred = self.predict(input_tensor)
            predictions.extend(next_pred[0].tolist()[:n_forward])
            idx += n_forward

        predictions = np.array(predictions[:len(true_values)])*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        true_values = true_values*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
        return predictions, true_values
    
    def get_metrics(self, stock_list, test_data, scalers, n_back, n_forward, day_ahead=None):
        model_metrics = pd.DataFrame()

        for stock in stock_list:
            x_test, y_test = get_model_inputs(stock, test_data, n_back, n_forward)
            y_test = y_test*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            y_test_pred = self.predict(x_test.reshape((x_test.shape[0], -1)))*np.sqrt(scalers[stock].var_[3]) + scalers[stock].mean_[3]
            
            rmse = np.sqrt(np.mean((y_test - y_test_pred)**2, axis=0))
            mae = np.mean(np.abs(y_test - y_test_pred), axis=0)
            mape = np.mean(np.abs(y_test - y_test_pred)/y_test, axis=0)

            if day_ahead is not None:
                rmse = rmse[day_ahead]
                mae = mae[day_ahead]
                mape = mape[day_ahead]
            else:
                rmse = np.mean(rmse)
                mae = np.mean(mae)
                mape = np.mean(mape)

            metrics = pd.DataFrame({'stock': [stock], 'rmse': [rmse], 'mae': [mae], 'mape': [mape]})
            model_metrics = pd.concat([model_metrics, metrics], axis=0)

        model_metrics = model_metrics.reset_index(drop=True)  
        return model_metrics
    
