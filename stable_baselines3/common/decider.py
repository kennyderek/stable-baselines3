
import torch.nn as nn

class Decider(nn.Module):

    def __init__(self, input_size, output_size):
        super(Decider, self).__init__()

        self.batch_size = 10 # 10 trajectories
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=64)
        self.fc1 = nn.Linear(64, output_size) # output_size is size of the context
    
    def forward(self, sequence):
        '''
        squence: (seq_len, batch, input_size)
        return: output, (h_n, c_n)
                where output is (seq_len, batch, num_directions * hidden_size)
        '''

        output, hiddens = self.rnn(sequence) # returns a packed_padded_sequence
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(output) # (seq_len, batch, num_directions * hidden_size), (h_n, c_n)
        logits = self.fc1(unpacked[-1]) # (batch, output_size)
        return logits

class FCNDecider(nn.Module):

    def __init__(self, input_size, output_size):
        super(FCNDecider, self).__init__()

        self.batch_size = 10 # 10 trajectories
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, sequence):
        '''
        squence: (batch, input_size)
        return: batch, output_size
        '''

        # output, hiddens = self.rnn(sequence) # returns a packed_padded_sequence
        # unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(output) # (seq_len, batch, num_directions * hidden_size), (h_n, c_n)
        # logits = self.fc1(unpacked[-1].double()) # (batch, output_size)
        logits = self.fc(sequence)
        return logits
