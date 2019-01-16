
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import time

from car_model_code import car_model

import os
import sys
import glob

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 7
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        # print (x.shape)
        out = self.conv1(x)
        out = self.relu1(out)
        # print (out.shape)
        out = self.conv2(out)
        out = self.relu2(out)
        # print (out.shape)
        out = self.conv3(out)
        out = self.relu3(out)
        # print (out.shape)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
    # print (out.shape)
        out = self.fc5(out)
        # print (out.shape)
        

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[:,180:-180]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)

    cv2.imwrite('bgr_frame.png',image_data)
    # image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))

    return image_data

def save_checkpoint(state,iteration):
    print ("saving to = pretrained_model/current_model_" + str(iteration) + ".pth.tar" )
    torch.save(state, "pretrained_model/current_model_" + str(iteration) + ".pth.tar")
    #torch.save(state, filename)

def train(model, start,resume_path):
    iteration_flag = 0
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    iteration = 0
    epsilon = model.initial_epsilon

    if (resume_path != '' ):
        print ("loading from =", resume_path)
        checkpoint = torch.load(resume_path)
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epsilon = checkpoint['epsilon']
    # initialize mean squared error loss`
    criterion = nn.MSELoss()

    # instantiate game
    game_state = car_model()  # start the game 

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)  # according to the state you take some action and get some reward
    

    image_data = resize_and_bgr2gray(image_data)
    prathi_state = [image_data, image_data, image_data, image_data]
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)  # find the next state
    # print (state.shape)
    # exit()
    # initialize epsilon value
   	
    

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations: # repeat the process 
        iteration_flag = iteration_flag + 1
        # get output from the neural network
        output = model(state)[0]
        # exit()
        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        
        
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]*3

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()


        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            save_checkpoint({
                'iteration':  iteration,
                'epsilon' : epsilon,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                #'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),}, iteration)

        # if iteration % 500 == 0:
        #     torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()), "loss:", loss)


def test(model):
    game_state = car_model()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    exit()
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def find_latest_model():
    find_latest_model = glob.glob('pretrained_model/current_model_*.pth.tar')

    if (find_latest_model == []):
        return ''
    else:
        find_latest_model = find_latest_model
        iteration_list = [int(i.split('_')[-1].split('.')[0]) for i in find_latest_model]
        return 'pretrained_model/current_model_'+str(max(iteration_list))+'.pth.tar' 

def main(mode):

    if mode == 'test':
        model = NeuralNetwork()
        if torch.cuda.is_available():
            # put on GPU if CUDA is available
            model = model.cuda()
        checkpoint = torch.load('/home/hemanth/PyCar/iteration_1_models/current_model_200000.pth.tar')
      	
        model.load_state_dict(checkpoint['state_dict'])
        test(model)
    elif mode == 'train':
    	
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        model = NeuralNetwork()
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            model = model.cuda()
        model.apply(init_weights)
        start = time.time()
        resume_path = find_latest_model()
        train(model, start, resume_path)


if __name__ == "__main__":
    main(sys.argv[1])

