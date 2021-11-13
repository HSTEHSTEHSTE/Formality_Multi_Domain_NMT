from transformers import AutoModel, AutoTokenizer
import torch
import model
import os
import pandas as pd
import tqdm

bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

# Hyper parameters
batch_size = 128
dev_batch_size = 16
max_iterations = 100
initial_learning_rate = .001
lr_decay = .5
print_every = 2

# Load data
data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/data.csv"), header=None, index_col=None, delimiter=',')
assert data_array.shape[1] == 2
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array.loc[:dev_size]
test_data_array = data_array.loc[dev_size:dev_size + test_size]
train_data_array = data_array.loc[dev_size + test_size:]

# Initialise model
classifier = model.LinearDecoder(768, 2) # Magic numbers: 768 length of pre-trained BERT output; 2: number of formality labels
criterion = torch.nn.NLLLoss()
optimiser = torch.optim.Adam(classifier.parameters(), lr=initial_learning_rate, weight_decay=0)
previous_loss = None
lr = initial_learning_rate

accuracies = []
losses = []
for iteration_number in range(0, max_iterations):
    batch_array = train_data_array.sample(n=batch_size)
    output_pooled_list = []
    for sentence_label_pair in tqdm.tqdm(batch_array.iterrows(), total=batch_size):
        sentence = sentence_label_pair[1].iloc[0][:510] # magic number 512: pre-trained BERT length limit
        inputs = tokenizer(sentence, return_tensors="pt")
        output_pooled = bertjapanese(**inputs).pooler_output # shape [1, 768]
        output_pooled_list.append(output_pooled)
    target_labels = torch.tensor(batch_array.iloc[:, [1]].to_numpy()).squeeze(1) # shape [batch_size]

    classifier_input_batched = torch.stack(output_pooled_list, dim=0) # shape [batch_size, 768]
    classifier.train()

    optimiser.zero_grad()
    
    # main forward pass
    output_labels = classifier(classifier_input_batched).squeeze(1) # shape [batch_size, 2]
    loss = criterion(output_labels, target_labels)
    loss.backward()
    optimiser.step()

    # calculate training accuracy
    output_labels = torch.argmax(output_labels, dim=1)
    accuracy_tensor = torch.where(output_labels - target_labels < .5, torch.tensor(1), torch.tensor(0))
    accuracy = torch.sum(accuracy_tensor) / batch_size
    accuracies.append(accuracy)
    losses.append(loss.item())

    # calculate dev loss, update learning rate
    if (iteration_number + 1) % print_every == 0:
        print("Iteration ", iteration_number + 1, " , loss is ", loss.item(), " , training accuracy is ", accuracy.item())

        dev_batch_array = dev_data_array.sample(n=dev_batch_size)
        dev_output_pooled_list = []
        for sentence_label_pair in tqdm.tqdm(dev_batch_array.iterrows(), total=dev_batch_size):
            sentence = sentence_label_pair[1].iloc[0][:510] # magic number 512: pre-trained BERT length limit
            inputs = tokenizer(sentence, return_tensors="pt")
            output_pooled = bertjapanese(**inputs).pooler_output # shape [1, 768]
            dev_output_pooled_list.append(output_pooled)
        dev_target_labels = torch.tensor(dev_batch_array.iloc[:, [1]].to_numpy()).squeeze(1) # shape [dev_batch_size]

        dev_classifier_input_batched = torch.stack(dev_output_pooled_list, dim=0) # shape [dev_batch_size, 768]
        classifier.eval()

        optimiser.zero_grad()
        dev_output_labels = classifier(dev_classifier_input_batched).squeeze(1) # shape [dev_batch_size, 2]
        
        # update learning rate
        dev_loss = criterion(dev_output_labels, dev_target_labels)
        if previous_loss is None:
            previous_loss = dev_loss
        elif previous_loss < dev_loss:
            lr_new = lr * lr_decay
            print("Dev loss increased. Reducing learning rate from ", lr, " to ", lr_new)
            lr = lr_new
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr