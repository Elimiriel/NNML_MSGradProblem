import torch
import torch.nn as nn
import numpy as np


class moving_avg(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, 
                                padding=0)
        
    def forward(self, x: np.ndarray|torch.Tensor):
        # 차원 보정
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"moving_avg expects 3D input, got shape: {x.shape}")
        # 앞/뒤 패딩 생성
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 평균
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size: int):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        
    def forward(self, x: np.ndarray|torch.Tensor):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual 
        

class LTSF_DLinear(nn.Module):
    """DLinear model for time series forecasting"""
    def __init__(self, 
                 window_size: int, forcast_size: int, 
                 kernel_size: int, individual: bool, 
                 input_size: int, output_size: int):
        """DLinear model for time series forecasting

        Args:
            window_size (int): numbers of input range in time, including current
            forcast_size (int): numbers of forecast in time
            kernel_size (int): size of moving average kernel
            individual (bool): decides trend and seasonal decomposition to use individual linear layers or not
            input_size (int): numbers of input features
            output_size (int): numbers of output features
        """
        super().__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.input_size = input_size
        self.output_size = output_size
        
        if self.individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(window_size, forcast_size)
                for _ in range(self.output_size)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(window_size, forcast_size)
                for _ in range(self.output_size)
            ])
        else:
            self.Linear_Trend = nn.Linear(window_size, forcast_size)
            self.Linear_Seasonal = nn.Linear(window_size, forcast_size)
    
    def forward(self, x: np.ndarray|torch.Tensor):
        # x: [B, T, input_size] → slice to output_size only
        if x.shape[2] > self.output_size:
            x = x[:, :, :self.output_size]
            
        B, T, C = x.shape
        trend_init, seasonal_init = self.decompsition(x)  # [B, T, C]
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)  # [B, C, T]
        
        if self.individual:
            trend_output = torch.zeros([B, C, self.forcast_size], dtype=x.dtype, device=x.device)
            seasonal_output = torch.zeros_like(trend_output)
            for idx in range(C):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            
        return (seasonal_output + trend_output).permute(0, 2, 1)  # [B, F, C]
    

class LTSF_NLinear(nn.Module):
    """NLinear model for time series forecasting"""
    def __init__(self, window_size: int, forcast_size: int, individual: bool, input_size: int, output_size: int):
        """NLinear model for time series forecasting

        Args:
            window_size (int): numbers of input range in time, including current
            forcast_size (int): numbers of forecast in time
            individual (bool): decides trend and seasonal decomposition to use individual linear layers or not
            input_size (int): numbers of input features
            output_size (int): numbers of output features
        """
        super().__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.input_size = input_size
        self.output_size = output_size

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(window_size, forcast_size)
                for _ in range(self.output_size)
            ])
        else:
            self.Linear = nn.Linear(window_size, forcast_size)
            
    def forward(self, x: np.ndarray|torch.Tensor):
        # x: [B, T, input_size] → slice to output_size only
        if x.shape[2] > self.output_size:
            x = x[:, :, :self.output_size]  # example case: only 1 feature
        seq_last = x[:, -1:, :].detach()  # [B, 1, output_size]
        x = x - seq_last
        
        if self.individual:
            B, T, C = x.shape  # C == output_size
            output = torch.zeros([B, self.forcast_size, C], 
                                 dtype=x.dtype, device=x.device)
            for i in range(C):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        x = x + seq_last
        return x  # shape: [B, forcast_size, output_size]
    

class LSTMModel(nn.Module):
    """Long Short Term Memory Model for time series forecasting"""
    def __init__(self, input_size: int, 
                 hidden_size: int, num_layers: int, 
                 forecast_size: int = 1, output_size: int = 1, 
                 bidirectional: bool = False):
        """Long Short Term Memory Model for time series forecasting

        Args:
            input_size (int): number of input features
            hidden_size (int): number of hidden units per LSTM layer
            num_layers (int): numbers of LSTM layers
            forecast_size (int, optional): number of time steps to forecast. Defaults to 1.
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
            bidirectional (bool, optional): decides to use bidirectional LSTM(reduce forget long-term dependencies) or not. Defaults to False.
        """
        super().__init__()
        self.forecast_size = forecast_size
        self.output_size = output_size
        self.bilstm = nn.LSTM(input_size, hidden_size, 
                              num_layers, batch_first=True, 
                              bidirectional=bidirectional)

        direction_mult = 2 if bidirectional else 1
        self.fc = nn.Linear(direction_mult * hidden_size, 
                            forecast_size * output_size)

    def forward(self, x: np.ndarray|torch.Tensor):
        out, _ = self.bilstm(x)  # [B, T, H]
        last_hidden = out[:, -1, :]  # [B, H*D]
        proj = self.fc(last_hidden)  # [B, F*O]
        proj = proj.view(-1, self.forecast_size, self.output_size)  # [B, F, O]
        return proj
    
    
class GRUModel(nn.Module):
    """Gated Recurrent Unit Model for time series forecasting, reduced from LSTM"""
    def __init__(self, input_size: int, 
                 hidden_size: int, num_layers: int, 
                 forecast_size: int = 1, output_size: int = 1, 
                 bidirectional: bool = False):
        """Gated Recurrent Unit Model for time series forecasting, reduced from LSTM

        Args:
            input_size (int): number of input features
            hidden_size (int): number of hidden units per GRU layer
            num_layers (int): numbers of GRU layers
            forecast_size (int, optional): number of time steps to forecast. Defaults to 1.
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
            bidirectional (bool, optional): decides to use bidirectional GRU(reduce forget long-term dependencies) or not. Defaults to False.
        """
        super(GRUModel, self).__init__()
        self.forecast_size = forecast_size
        self.output_size = output_size
        self.bigru = nn.GRU(input_size, 
                            hidden_size, num_layers, 
                            batch_first=True, bidirectional=bidirectional)
        direction_mult = 2 if bidirectional else 1
        self.fc = nn.Linear(direction_mult * hidden_size, 
                            forecast_size * output_size)

    def forward(self, x: np.ndarray|torch.Tensor):
        out, _ = self.bigru(x)
        last_hidden = out[:, -1, :]
        proj = self.fc(last_hidden)  # [B, F*O]
        proj = proj.view(-1, self.forecast_size, self.output_size)
        return proj
    

class CNNBiLSTMModel(nn.Module):
    """CNN layers+BiLSTM layers+FC layers for time series forecasting
    """
    def __init__(self, input_size: int, 
                 hidden_size: int, num_layers: int, 
                 forecast_size: int = 1, output_size: int = 1, 
                 kernel_size: int = 3):
        """CNN layers+BiLSTM layers+FC layers for time series forecasting

        Args:
            input_size (int): number of input features
            hidden_size (int): number of hidden units per LSTM layer
            num_layers (int): numbers of LSTM layers
            forecast_size (int, optional): number of time steps to forecast. Defaults to 1.
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
            kernel_size (int, optional): size of CNNs' kernel. Defaults to 3.
        """
        super(CNNBiLSTMModel, self).__init__()
        self.forecast_size = forecast_size
        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, 64, 
                               kernel_size=kernel_size, padding=1)  # 1st CNN
        self.conv2 = nn.Conv1d(64, 128, 
                               kernel_size=kernel_size, padding=1)  # 2nd CNN
        self.pool = nn.MaxPool1d(2)  # pooling
        self.lstm = nn.LSTM(128, 
                            hidden_size, num_layers, 
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, forecast_size * output_size)  # FC layer for output

    def forward(self, x: np.ndarray|torch.Tensor):
        # CNNs (input: batch_size*input_channels*seq_len)
        x = x.transpose(1, 2)  # dim to Conv1d(batch_size*input_channels*seq_len)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        # LSTM adjusment (batch_size x seq_len x feature_size)
        x = x.transpose(1, 2)
        # BiLSTM
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        # FC
        proj = self.fc(last_hidden)  # [B, F*O]
        proj = proj.view(-1, self.forecast_size, self.output_size)
        return proj
    

class CNNBiGRUModel(nn.Module):
    """CNN layers+BiGRU layers+FC layers for time series forecasting
    """
    def __init__(self, 
                 input_size: int, hidden_size: int, 
                 num_layers: int, forecast_size: int = 1, 
                 output_size: int = 1, kernel_size: int = 3):
        """CNN layers+BiGRU layers+FC layers for time series forecasting

        Args:
            input_size (int): number of input features
            hidden_size (int): number of hidden units per GRU layer
            num_layers (int): numbers of GRU layers
            forecast_size (int, optional): number of time steps to forecast. Defaults to 1.
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
            kernel_size (int, optional): size of CNNs' kernel. Defaults to 3.
        """
        super(CNNBiGRUModel, self).__init__()
        self.forecast_size = forecast_size
        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, 64, 
                               kernel_size=kernel_size, padding=1)  # CNN 1
        self.conv2 = nn.Conv1d(64, 128, 
                               kernel_size=kernel_size, padding=1)  # CNN 2
        self.pool = nn.MaxPool1d(2)  # pooling
        self.bigru = nn.GRU(128, hidden_size, num_layers, 
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, forecast_size * output_size)  # FC layer
        
    def forward(self, x: np.ndarray|torch.Tensor):
        # CNN(input: batch_size*input_channels*seq_len)
        x = x.transpose(1, 2)  # Conv1d fit (batch_size*input_channels*seq_len)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        # GRU adjusment (batch_size x seq_len x feature_size)
        x = x.transpose(1, 2)
        # BiGRU
        out, _ = self.bigru(x)
        last_hidden = out[:, -1, :]
        # FC
        proj = self.fc(last_hidden)  # [B, F*O]
        proj = proj.view(-1, self.forecast_size, self.output_size)
        return proj
    

class CNNBiGRUBiLSTMModel(nn.Module):
    """CNN layers+BiGRU layers+BiLSTM layers+FC layers for time series forecasting"""
    def __init__(self, 
                 input_size: int, hidden_size: int, num_layers: int, 
                 forecast_size: int = 1, output_size: int = 1, 
                 kernel_size: int = 3):
        """CNN layers+BiGRU layers+BiLSTM layers+FC layers for time series forecasting

        Args:
            input_size (int): number of input features
            hidden_size (int): number of hidden units per GRU and LSTM layer
            num_layers (int): numbers of GRU and LSTM layers
            forecast_size (int, optional): number of time steps to forecast. Defaults to 1.
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
            kernel_size (int, optional): size of CNNs' kernel. Defaults to 3.
        """
        super(CNNBiGRUBiLSTMModel, self).__init__()
        self.forecast_size = forecast_size
        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, 64, 
                               kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 
                               kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gru = nn.GRU(128, hidden_size, num_layers, 
                          bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(2*hidden_size, hidden_size, 
                            num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, forecast_size * output_size)  # modified to multiple forecastings
        
    def forward(self, x: np.ndarray|torch.Tensor):
        # CNN(input: batch_size x input_channels x seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        # BiLSTM+GRU(batch_size x seq_len x feature_size)
        x = x.transpose(1, 2)
        # BiGRU-BiLSTM
        out, _ = self.gru(x)
        out, _ = self.lstm(out)
        last_hidden = out[:, -1, :]
        proj = self.fc(last_hidden)  # [B, F*O]
        proj = proj.view(-1, self.forecast_size, self.output_size)  # [B, F, O]
        return proj
    
    
class NBeatsBlock(nn.Module):
    """N B E A T S
    """
    def __init__(self, 
                input_size: int, hidden_size: int, 
                forecast_size: int, output_size: int = 1, 
                num_layers: int = 4):
        super().__init__()
        self.output_size = output_size
        self.fc_stack = nn.Sequential(*[
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, forecast_size * output_size)
        
    def forward(self, x: np.ndarray|torch.Tensor):
        B, T, C = x.shape
        x = x.reshape(B, -1)  # flatten: [B, T*C]
        x = self.fc_stack(x)
        x = self.fc_out(x)  # [B, F*O]
        return x.view(B, -1, self.output_size)  # [B, F, O]
    

class NHiTSBlock(nn.Module):
    """NHiTS
    """
    def __init__(self, 
                 input_size: int, hidden_size: int, 
                 forecast_size: int, output_size: int = 1, 
                 kernel_size: int = 3, num_layers: int = 3):
        super().__init__()
        self.output_size = output_size
        self.conv_stack = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(input_size, 
                                    hidden_size, kernel_size, 
                                    padding=1), nn.ReLU())
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, forecast_size * output_size)
        
    def forward(self, x: np.ndarray|torch.Tensor):
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_stack(x)  # [B, H, T]
        x = torch.mean(x, dim=-1)  # Global average pooling [B, H]
        x = self.fc_out(x)  # [B, F*O]
        return x.view(x.shape[0], -1, self.output_size)  # [B, F, O]


class TransformerForecast(nn.Module):
    """Transformer model(Attention-based AutoEncoder) for multimodal timeseries forecasting
    """
    def __init__(self, 
                 input_size: int, 
                 d_model: int, nhead: int, 
                 num_layers: int, forecast_size: int, 
                 output_size: int = 1):
        """Transformer model(Attention-based AutoEncoder) for multimodal timeseries forecasting

        Args:
            input_size (int): number of input features
            d_model (int): dimension of model
            nhead (int): number of heads in multiheadattention
            num_layers (int): number of encoder layers
            forecast_size (int): number of time steps to forecast
            output_size (int, optional): feature numbers to forecast. Defaults to 1.
        """
        super().__init__()
        self.output_size = output_size
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 
                                             num_layers=num_layers)
        self.fc = nn.Linear(d_model, forecast_size * output_size)
        
    def forward(self, x: np.ndarray|torch.Tensor):
        x = self.embedding(x)  # [B, T, d_model]
        x = self.encoder(x)    # [B, T, d_model]
        x = x[:, -1, :]        # 마지막 타임스텝만 사용
        x = self.fc(x)         # [B, F*O]
        return x.view(x.shape[0], -1, self.output_size)  # [B, F, O]
    

def create_sliding_window(X, y, y_index, 
                          window_size: int, forecast_size: int = 1):
    """sliding window generator for singlemodal timeseries forecasting

    Args:
        X (_type_): independent variables
        y (_type_): dependent variables
        y_index (_type_): column names of dependent variables
        window_size (int): numbers of input range in time
        forecast_size (int, optional): number of time steps to forecast. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, y_index
    """
    X_sliding, y_sliding, y_sliding_index = [], [], []
    
    for i in range(len(X) - window_size - forecast_size + 1):
        X_sliding.append(X[i: i + window_size])
        y_sliding.append(y[i + window_size: i+window_size+forecast_size])
        y_sliding_index.append(y_index[i + window_size+forecast_size-1])
    
    return np.array(X_sliding), np.array(y_sliding), np.array(y_sliding_index)


def create_sliding_window_flatten(X, y, y_index, 
                                  window_size: int, forecast_size: int = 1):
    """sliding window generator for multimodal timeseries forecasting

    Args:
        X (_type_): independent variables
        y (_type_): dependent variables
        y_index (_type_): column names of dependent variables
        window_size (int): numbers of input range in time
        forecast_size (int, optional): number of time steps to forecast. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, y_index
    """
    X_sliding, y_sliding, y_sliding_index = [], [], []
    
    for i in range(len(X) - window_size - forecast_size + 1):
        X_sliding.append(X[i: i + window_size].flatten())  # merge timesteps
        y_sliding.append(y[i + window_size:i+window_size+forecast_size])
        y_sliding_index.append(y_index[i + window_size+forecast_size-1])
    
    return np.array(X_sliding), np.array(y_sliding), np.array(y_sliding_index)