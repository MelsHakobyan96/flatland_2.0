import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math



def spatial_pyramid_pool(previous_conv, levels, mode):
    """
    Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
    (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
    :param previous_conv input tensor of the previous convolutional layer
    :param levels defines the different divisions to be made in the width and height dimension
    :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
    :returns a tensor vector with shape [batch x 1 x n],
                                        where n: sum(filter_amount*level*level) for each level in levels
                                        which is the concentration of multi-level pooling



    credit for this method: Marc A., github: revidee
    """
    num_sample = previous_conv.size(0)
    previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
    for i in range(len(levels)):
        h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
        w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
        w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
        w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
        h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
        h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
        assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
               h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

        padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                             mode='constant', value=0)
        if mode == "max":
            pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
        elif mode == "avg":
            pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
        else:
            raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
        x = pool(padded_input)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)

    return spp



class CNN_RNN(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, hidden_size, bidirectional, rnn_num_layers, action_size):
		super(CNN_RNN, self).__init__()

		self.hidden_size = hidden_size
		self.rnn_input_size = 1100
		self.bidirectional = bidirectional
		self.rnn_num_layers = rnn_num_layers
		self.levels = [1, 2, 3, 6]



		self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)
		self.batchnorm1 = nn.BatchNorm2d(num_features = out_channels)

		self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels*4, kernel_size = kernel_size)
		self.batchnorm2 = nn.BatchNorm2d(num_features = out_channels*4)

		self.conv3 = nn.Conv2d(in_channels = out_channels*4, out_channels = out_channels*8, kernel_size = kernel_size)
		self.batchnorm3 = nn.BatchNorm2d(num_features = out_channels*8)

		self.conv4 = nn.Conv2d(in_channels = out_channels*8, out_channels = out_channels*4, kernel_size = kernel_size)
		self.batchnorm4 = nn.BatchNorm2d(num_features = out_channels*4)

		self.conv5 = nn.Conv2d(in_channels = out_channels*4, out_channels = out_channels, kernel_size = kernel_size)
		self.batchnorm5 = nn.BatchNorm2d(num_features = out_channels)

		self.lstm = nn.LSTM(self.rnn_input_size, hidden_size, num_layers=rnn_num_layers, bidirectional=bidirectional)
		self.actor_head_linear = nn.Linear(hidden_size*(1+int(self.bidirectional)), action_size)
		self.actor_head_final = nn.Softmax(dim=-1)
		self.critic_head = nn.Linear(hidden_size*(1+int(self.bidirectional)), 1)





	def forward(self, inputs):
		"""
		inputs is in shape of (batch_size, agent_num, channel_num, map_width, map_height)
		"""
		batch_size = inputs.size(0)
		agent_num = inputs.size(1)

		# reduce the dimension of the input to fit the CNN
		inputs = inputs.view(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))

		conved_1 = self.conv1(inputs)
		batched_1 = self.batchnorm1(conved_1)
		leaky_relued_1 = F.leaky_relu(batched_1)

		conved_2 = self.conv2(leaky_relued_1)
		batched_2 = self.batchnorm2(conved_2)
		leaky_relued_2 = F.leaky_relu(batched_2)

		conved_3 = self.conv3(leaky_relued_2)
		batched_3 = self.batchnorm3(conved_3)
		leaky_relued_3 = F.leaky_relu(batched_3)

		conved_4 = self.conv4(leaky_relued_3)
		batched_4 = self.batchnorm4(conved_4)
		leaky_relued_4 = F.leaky_relu(batched_4)

		conved_5 = self.conv5(leaky_relued_4)
		batched_5 = self.batchnorm5(conved_5)
		leaky_relued_5 = F.leaky_relu(batched_5)
		pooled_leaky_relued_5 = spatial_pyramid_pool(previous_conv = leaky_relued_5, levels = self.levels, mode = 'avg')
		pooled_leaky_relued_5 = pooled_leaky_relued_5.view(pooled_leaky_relued_5.size()[0], 1, pooled_leaky_relued_5[0].size()[0])

		if batch_size > 1:
			# split back the tensor to fit into LSTM
			pooled_leaky_relued_5 = torch.stack(pooled_leaky_relued_5.split(batch_size))
			pooled_leaky_relued_5 = pooled_leaky_relued_5.squeeze()

		if agent_num == 1:
			pooled_leaky_relued_5 = pooled_leaky_relued_5.view(1, batch_size, self.rnn_input_size)


		output, hidden = self.lstm(pooled_leaky_relued_5, self.init_hidden(batch_size))
		value = self.critic_head(output)
		policy = self.actor_head_linear(output)
		policy = self.actor_head_final(policy)

		return value, policy



	def init_hidden(self, batch_size):
	    weight = next(self.parameters()).data
	    return (Variable(weight.new(self.rnn_num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size).zero_()),
	           Variable(weight.new(self.rnn_num_layers * (1 + int(self.bidirectional)), batch_size, self.hidden_size).zero_()))



# conv = CNN_RNN(in_channels = 8, out_channels = 5, kernel_size = 1, hidden_size = 4, bidirectional = True, rnn_num_layers = 1, action_size = 5)
# x = torch.rand((5, 1, 8, 21, 21))
# # x = torch.FloatTensor(x)
# print(conv.forward(x))


