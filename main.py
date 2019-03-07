import torch
import random
from preprocess import loadPrepareData, batch2TrainData
from model import EncoderRNN
from loss import am_softmax_loss
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def train(training_batch, embedding, encoder, optimizer, clip_grad, criterion, batch_size, device):
    input_, _, input_len, _, output_ = training_batch
    optimizer.zero_grad()
    
    # User for am_softmax_loss
    one_hot_output_ = torch.zeros(output_.shape[0], class_num).scatter_(1, output_.reshape(output_.shape[0],1),1)

    input_variable  = input_.to(device)
    len_variable    = input_len.to(device)

    output_variable = output_.to(device)
    one_hot_output_variable = one_hot_output_.to(device)

    logits_,_ = encoder(input_variable, len_variable)
    loss = criterion(logits_, one_hot_output_variable, output_variable) 
    return loss

def evaluate(val_batch, embedding, encoder, batch_size, device):
    input_, _, input_len, _, output_ = val_batch
    input_variable  = input_.to(device)
    len_variable    = input_len.to(device)
    output_variable = output_.to(device)
    
    encoder.eval()
    logits_, _ = encoder(input_variable, len_variable)
    _,predict = torch.max(logits_, 1)
    
    num = 0
    for i in range(len(output_)):
	    if output_[i] == predict.cpu()[i]:
		    num += 1
    return num * 1.0 / len(output_)

def get_val_test_pairs(voc, val_test_pairs):    
    val_test_batches = []
    batch = []    
    for i in tqdm( range(len(val_test_pairs)) ):
       if len(batch) != batch_size:
          batch.append(val_test_pairs[i])
       else:
          val_test_batch_data = batch2TrainData(voc,batch)
          _,_,val_input_len,_,_ = val_test_batch_data
          if sorted(val_input_len.tolist(),reverse=True) != val_input_len.tolist():
             batch = []
             continue
          val_test_batches.append(val_test_batch_data)
          batch = []
    return val_test_batches

def verification(test_batches, val_batches, topn, embedding, encoder, batch_size, device):
    topn = 2
    hiddens = []
    real_labels = []
    encoder.eval()
    for i in range(len(val_batches)):
        input_, _, input_len, _, output_ = val_batches[i]
        input_variable  = input_.to(device)
        len_variable    = input_len.to(device)
        output_variable = output_.to(device)
        _, hidden = encoder(input_variable, len_variable)
        hiddens.append(hidden)
        real_labels.append(output_)
    hiddens_list = [hiddens[i].data.cpu().numpy() for i in range(len(hiddens))]
    label_list = [real_labels[i].data.cpu().numpy() for i in range(len(real_labels))]
    tensor_hiddens_list = torch.tensor(hiddens_list).reshape(batch_size * len(hiddens_list),-1)
    tensor_label_list = torch.tensor(label_list).reshape(batch_size * len(label_list),-1)
    tensor_hiddens_list = tensor_hiddens_list.to(device)
    
    acc_num = 0
    for j in range(len(test_batches)):
        input_, _, input_len, _, output_ = test_batches[i]
        input_variable  = input_.to(device)
        len_variable    = input_len.to(device)
        output_variable = output_.to(device)
        encoder.eval()
        _, hidden = encoder(input_variable, len_variable)
        for k in range(hidden.shape[0]):
            diff_feature = tensor_hiddens_list - hidden[k]
            score_list = (diff_feature.mul(diff_feature).sum(1)) / diff_feature.shape[1]  
            score_list = score_list.data.cpu().numpy().tolist()
            topn_score_list = sorted(score_list)[:topn]
            for score in topn_score_list:
                index = score_list.index(score)
                if tensor_label_list[index] == output_[k]:
                   acc_num += 1
                   break
    return acc_num * 1.0 / len(test_batches) * batch_size
    


if __name__ == '__main__':
    
    corpus_name         = 'iask'
    datafile            = '../data/data_iask_20000.csv'
    class_num           = 98611

    hidden_size         = 500
    encoder_n_layers    =  3
    dropout_ratio       = 0.1
    
    learning_rate       = 0.0001
    clip_grad           = 50
    batch_size          = 64
    n_iteration         = 100000
    print_interval	= 100 
    scale               = 30
    margin              = 0.35

    voc, train_pairs, val_pairs, test_pairs  = loadPrepareData(corpus_name, datafile)
    
    embedding   = torch.nn.Embedding(voc.num_words, hidden_size)
    
    encoder = EncoderRNN(hidden_size, embedding, class_num, encoder_n_layers, dropout_ratio)
    encoder = encoder.to(device)
    encoder.train()
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    
    standard_softmax_loss = torch.nn.CrossEntropyLoss()
    
    #criterion = standard_softmax_loss
    criterion = am_softmax_loss(standard_softmax_loss, class_num, scale, margin)

    print('Prepare data.\n')    
    training_batches = []
    val_batches = []
    test_batches = []

    for _ in tqdm( range(n_iteration) ):
        batch_data = batch2TrainData(voc,[random.choice(train_pairs) for _ in range(batch_size)])
        _,_,input_len,_,_ = batch_data
        if sorted(input_len.tolist(),reverse=True) != input_len.tolist():
            continue
        training_batches.append(batch_data)
    
    val_batches = get_val_test_pairs(voc, val_pairs)    
    test_batches = get_val_test_pairs(voc, test_pairs)    
    print('Start training.\n')
    for iteration in range(n_iteration):
        training_batch = training_batches[iteration]
        loss = train(training_batch, embedding, encoder, optimizer, clip_grad, criterion, batch_size, device)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
        optimizer.step()
        print('iteration: {}/{}, loss: {:.5f}'.format(iteration, n_iteration, loss))

        if iteration % print_interval == 0:
               acc = 0
               for i in range(len(val_batches)):
                   val_batch = val_batches[i]
                   acc_batch = evaluate(val_batch, embedding, encoder, batch_size, device)
                   acc += acc_batch
               encoder.train()
               print('iteration: {}/{}, classification acc: {:.2f}'.format(iteration, n_iteration, acc/len(val_batches) ))
        if iteration % (print_interval * 2) == 0:
                acc = verification(test_batches, val_batches, 2, embedding, encoder, batch_size, device)
                encoder.train()
                print('iteration: {}/{}, verification acc: {:.3f}'.format(iteration, n_iteration, acc))	
