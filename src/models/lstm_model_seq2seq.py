import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, inputsize=1, hiddensize=64, numlayers=2, dropout=0.2):  # type: ignore
        super(Encoder, self).__init__()  # type: ignore
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.lstm = nn.LSTM(
            input_size=inputsize,  # type: ignore
            hidden_size=hiddensize,  # type: ignore
            num_layers=numlayers,  # type: ignore
            batch_first=True,  # type: ignore
            dropout=dropout  # type: ignore
        )
    def forward(self, x):  # type: ignore
        # x: (batchsize, seqlen, inputsize)
        outputs, (hidden, cell) = self.lstm(x)  # type: ignore
        return hidden, cell

class Decoder(nn.Module):
    def __init__(  # type: ignore
        self, inputsize=1, hiddensize=64, numlayers=2, outputsize=1, dropout=0.2
    ):
        super(Decoder, self).__init__()  # type: ignore
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.lstm = nn.LSTM(
            input_size=inputsize,  # type: ignore
            hidden_size=hiddensize,  # type: ignore
            num_layers=numlayers,  # type: ignore
            batch_first=True,  # type: ignore
            dropout=dropout  # type: ignore
        )
        self.fc = nn.Linear(hiddensize, outputsize)  # type: ignore
    def forward(self, x, hidden, cell):  # type: ignore
        # x: (batchsize, 1, inputsize)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))  # type: ignore
        prediction = self.fc(output[:, -1, :])  # type: ignore  # (batchsize, outputsize)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, forecastlen):  # type: ignore
        super(Seq2Seq, self).__init__()  # type: ignore
        self.encoder = encoder
        self.decoder = decoder
        self.forecastlen = forecastlen
    def forward(self, src):  # type: ignore
        src.size(0)  # type: ignore
        src.size(2)  # type: ignore
        # Encode input sequence
        hidden, cell = self.encoder(src)  # type: ignore
        # Initialize decoder input with last value of input sequence
        decoderinput = src[:, -1:, :]  # type: ignore  # shape: (batchsize, 1, inputsize)
        outputs = []
        for _ in range(self.forecastlen):
            out, hidden, cell = self.decoder(decoderinput, hidden, cell)  # type: ignore
            outputs.append(out.unsqueeze(1))  # type: ignore  # shape: (batchsize, 1, outputsize)
            decoderinput = out.unsqueeze(1)  # type: ignore  # use predicted value as next input
        outputs = torch.cat(outputs, dim=1)  # type: ignore # shape: (batchsize, forecastlen, output_size)
        return outputs
