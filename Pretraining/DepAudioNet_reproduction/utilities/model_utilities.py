import torch
import torch.nn as nn


def create_tensor_data(x, cuda):
    """
    Converts the data from numpy arrays to torch tensors

    Inputs
        x: The input data
        cuda: Bool - Set to true if using the GPU

    Output
        x: Data converted to a tensor
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def calculate_loss(prediction, target, cw=None, gender=True):
    """
    With respect to the final layer of the model, calculate the loss of the
    model.

    Inputs
        prediction: The output of the model
        target: The relative label for the output of the model
        cw: torch.Tensor - The class weights for the dataset
        gender: bool set True if splitting data according to gender

    Output
        loss: The BCELoss or NLL_Loss
    """
    if gender:
        if target.shape[0] != cw.shape[0]:
            fem_nd_w, fem_d_w, male_nd_w, male_d_w = cw
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            two_ind = (target == 2).nonzero().reshape(-1)
            three_ind = (target == 3).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, fem_nd_w[0])
            class_weights.scatter_(0, one_ind, fem_d_w[0])
            class_weights.scatter_(0, two_ind, male_nd_w[0])
            class_weights.scatter_(0, three_ind, male_d_w[0])
            cw = class_weights.reshape(-1, 1)
        target = target % 2
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
    else:
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
        if target.shape[0] != cw.shape[0]:
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, cw[0])
            class_weights.scatter_(0, one_ind, cw[1])
            cw = class_weights.reshape(-1, 1)

    if prediction.dim() == 1:
        prediction = prediction.view(-1, 1)

    bceloss = nn.BCELoss(weight=cw)
    import pdb
    #pdb.set_trace()
    loss = bceloss(prediction, target.float().view(-1, 1))

    return loss

def calculate_IDL_loss(output_orig, output_aug,temperature):
    '''
    Pi_xi_hat_list = torch.empty(20)
    import pdb
    for i in range(output_orig.shape[0]):
        numerator = torch.exp(torch.dot(output_orig[i],output_aug[i])/temperature)
        denomerator = torch.sum(torch.exp(torch.matmul(output_aug,output_orig[i])/temperature))
        #for j in range(output_orig.shape[0]):
        #    denomerator += torch.log(torch.dot(output_orig[j],output_aug[i])/temperature)
        Pi_xi_hat = numerator / denomerator
        Pi_xi_hat_list[i] = Pi_xi_hat
    #Pi_xj_mat = torch.zeros([output_orig.shape[0],output_orig.shape[0]],requires_grad = True)
    #for i in range(output_orig.shape[0]):
    #    for j in range(output_orig.shape[0]):
    #        numerator = torch.log(torch.dot(output_orig[i],output_orig[j])/temperature)
            #for k in range(output_orig.shape[0]):
            #    denomerator += torch.log(torch.dot(output_orig[k],output_orig[j])/temperature)
    exp_i_xj_mat = torch.exp(torch.matmul(output_orig,torch.transpose(output_orig,0,1))/temperature)
    Pi_xj_mat = torch.empty([output_orig.shape[0],output_orig.shape[0]])
    for i in range(output_orig.shape[0]):
        for j in range(output_orig.shape[0]):
            Pi_xj_mat[i,j] = exp_i_xj_mat[i,i] / torch.sum(exp_i_xj_mat[i,:])
            
    Ji_list = torch.empty(output_orig.shape[0])
    for i in range(output_orig.shape[0]):
        term2 = torch.sum(torch.log(1-Pi_xj_mat[i,:])) - torch.log(1-Pi_xj_mat[i,i])
        Ji = -1 * torch.log(Pi_xi_hat_list[i]) - term2
        Ji_list[i] = Ji
        
    import pdb
    J = torch.sum(Ji_list)
    J = J / output_orig.shape[0]
    '''
    Pi_xi_hat_list = torch.empty(output_orig.shape[0])
    for i in range(output_orig.shape[0]):
        numerator = torch.exp(torch.dot(output_orig[i],output_aug[i])/temperature)#inner product of two vectors
        denomerator = torch.sum(torch.exp(torch.matmul(output_aug[i],torch.transpose(output_orig,0,1))/temperature))
        Pi_xi_hat = numerator / denomerator
        Pi_xi_hat_list[i] = Pi_xi_hat
    exp_i_xj_mat = torch.exp(torch.matmul(output_orig,torch.transpose(output_orig,0,1))/temperature)
    Pi_xj_mat = torch.empty([output_orig.shape[0],output_orig.shape[0]])
    for i in range(output_orig.shape[0]):
        for j in range(output_orig.shape[0]):
            Pi_xj_mat[i,j] = exp_i_xj_mat[i,j] / torch.sum(exp_i_xj_mat[:,j])
    Ji_list = torch.empty(output_orig.shape[0])
    for i in range(output_orig.shape[0]):
        term2 = torch.sum(torch.log(1-Pi_xj_mat[i,:])) - torch.log(1-Pi_xj_mat[i,i])
        Ji = -1 * torch.log(Pi_xi_hat_list[i]) - term2
        Ji_list[i] = Ji
    J = torch.sum(Ji_list)
    J = J / output_orig.shape[0]
    return J


