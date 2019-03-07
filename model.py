import torch
import torch.nn.functional as F

class EncoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, embedding, class_num, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
        self.out = torch.nn.Linear(2 * n_layers * hidden_size, class_num)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        try:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        except Exception as e:
            import pdb
            pdb.set_trace()
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        # Sum bidirectional GRU outputs
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        
        # Permute dim
        hidden  = hidden.permute(1,0,2)
        hidden = hidden.contiguous().view(hidden.size()[0], hidden.size()[1] * hidden.size()[2])
        # Linear layers
        logits = self.out(hidden)
        # Return output and final hidden state
        return logits, hidden

