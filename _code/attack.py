import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_training_mode`.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]
        self.device = next(model.parameters()).device

        self._attack_mode = 'default'
        self._targeted = False
        self._return_type = 'float'
        self._supported_mode = ['default']

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self._attack_mode

    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self._attack_mode = 'default'
        self._targeted = False
        print("Attack mode is changed to 'default.'")

    def set_mode_targeted_by_function(self, target_map_function=None):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda images, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = 'targeted'
        self._targeted = True
        self._target_map_function = target_map_function
        print("Attack mode is changed to 'targeted.'")

    def set_mode_targeted_least_likely(self, kth_min=1):
        r"""
        Set attack mode as targeted with least likely labels.
        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(least-likely)"
        self._targeted = True
        self._kth_min = kth_min
        self._target_map_function = self._get_least_likely_label
        print("Attack mode is changed to 'targeted(least-likely).'")

    def set_mode_targeted_random(self, n_classses=None):
        r"""
        Set attack mode as targeted with random labels.
        Arguments:
            num_classses (str): number of classes.

        """
        if "targeted" not in self._supported_mode:
            raise ValueError("Targeted mode is not supported.")

        self._attack_mode = "targeted(random)"
        self._targeted = True
        self._n_classses = n_classses
        self._target_map_function = self._get_random_target_label
        print("Attack mode is changed to 'targeted(random).'")

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.

        Arguments:
            type (str): 'float' or 'int'. (Default: 'float')

        .. note::
            If 'int' is used for the return type, the file size of 
            adversarial images can be reduced (about 1/4 for CIFAR10).
            However, if the attack originally outputs float adversarial images
            (e.g. using small step-size than 1/255), it might reduce the attack
            success rate of the attack.

        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")

    def set_training_mode(self, model_training=False, batchnorm_training=False, dropout_training=False):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_pred=False):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_pred (bool): True for saving predicted labels (Default: False)

        """            
        if save_path is not None:
            image_list = []
            label_list = []
            if save_pred:
                pre_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)

        given_training = self.model.training
        given_return_type = self._return_type
        self._return_type = 'float'

        for step, (images, labels) in enumerate(data_loader):
            start = time.time()
            adv_images = self.__call__(images, labels)

            batch_size = len(images)

            if verbose or return_verbose:
                with torch.no_grad():
                    if given_training:
                        self.model.eval()
                    outputs = self.model(adv_images)
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = (pred == labels.to(self.device))
                    correct += right_idx.sum()
                    end = time.time()
                    delta = (adv_images - images.to(self.device)).view(batch_size, -1)
                    l2_distance.append(torch.norm(delta[~right_idx], p=2, dim=1))

                    rob_acc = 100 * float(correct) / total
                    l2 = torch.cat(l2_distance).mean().item()
                    progress = (step+1)/total_batch*100
                    elapsed_time = end-start
                    if verbose:
                        self._save_print(progress, rob_acc, l2, elapsed_time, end='\r')

            if save_path is not None:
                if given_return_type == 'int':
                    adv_images = self._to_uint(adv_images.detach().cpu())
                    image_list.append(adv_images)
                else:
                    image_list.append(adv_images.detach().cpu())
                    
                label_list.append(labels.detach().cpu())
                if save_pred:
                    pre_list.append(pred.detach().cpu())
                
        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end='\n')

        if save_path is not None:
            image_list = torch.cat(image_list, 0)
            label_list = torch.cat(label_list, 0)
            if save_pred:
                pre_list = torch.cat(pre_list, 0)
                torch.save((image_list, label_list, pre_list), save_path)
            else:
                torch.save((image_list, label_list), save_path)
            print('- Save complete!')

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        print('- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t' \
              % (progress, rob_acc, l2, elapsed_time), end=end)

    def _get_target_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function:
            return self._target_map_function(images, labels)
        raise ValueError('Please define target_map_function.')

    def _get_least_likely_label(self, images, labels=None):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        if self._kth_min < 0:
            pos = outputs.shape[1] + self._kth_min + 1
        else:
            pos = self._kth_min
        _, target_labels = torch.kthvalue(outputs.data, pos)
        target_labels = target_labels.detach()
        return target_labels.long().to(self.device)

    def _get_random_target_label(self, images, labels=None):
        if self._n_classses is None:
            outputs = self.model(images)
            if labels is None:
                _, labels = torch.max(outputs, dim=1)
            n_classses = outputs.shape[-1]
        else:
            n_classses = self._n_classses

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = self.random_int(0, len(l))
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)
    
    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ['model', 'attack']

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info['attack_mode'] = self._attack_mode
        info['return_type'] = self._return_type

        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        given_training = self.model.training

        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if 'Dropout' in m.__class__.__name__:
                        m = m.eval()

        else:
            self.model.eval()

        images = self.forward(*input, **kwargs)

        if given_training:
            self.model.train()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class PGD_steps(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, max_steps=20, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.max_steps = max_steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        outputs_start = self.model(images)
        labels = torch.argmax(outputs_start, dim=1)
        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
       
        i=0
        stp_atks = [0 for _ in range(images.shape[0])]
        idxs = []
        while i<=self.max_steps:
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cls_pred = torch.argmax(outputs, dim=1)

            for idx in range(images.shape[0]):
                if cls_pred[idx]!= labels[idx]:
                    if idx not in idxs:
                        if i == self.max_steps:
                            stp_atks[idx] = self.max_steps
                            break
                        else:
                            stp_atks[idx] = i
                            idxs.append(idx)
            # if (cls_pred != labels) and store_step:
            #     stp_atk = i
            #     store_step = False
                # print(f'Steps: {stp_atk} \t {prev_pred.item()} -> {cls_pred.item()}', end="")

            # for i in range(data.shape[0]):
            #     if i in idxs:
            #         continue
            #     else:
            #         if pred[i]!=pred_og[i]:
            #             flip_freq[i] = r
            #             idxs.append(i)

            # if i == self.max_steps and (cls_pred != labels) and store_step:
            #     stp_atk = self.max_steps
            #     break ## Get out of while loop as we don't want to update the adv-image again
            
            # Calculate loss
            
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # if i == self.max_steps and (cls_pred != labels):
            #     stp_atk = self.max_steps
            i+=1
        return adv_images, stp_atks


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)
    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.
    """
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        print('img, lbl: ', images.shape, labels.shape)
        print('w ', w.shape)
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)
        print('Best L2 ', best_L2.shape)
        
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)
        
        i=0
        store_step = True
        stp_atk = 0

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)  # (B, C, H, W)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            # print(current_L2)
            L2_loss = current_L2.sum()
            print('L2 loss: ', L2_loss)
            outputs = self.model(adv_images) # (B, C)
            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss
            print(cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()
            
            if (pre != labels) and store_step:
                stp_atk = i
                store_step = False
                # print(f'Steps: {stp_atk} \t {prev_pred.item()} -> {cls_pred.item()}', end="")

            if i == self.steps and (pre != labels) and store_step:
                stp_atk = self.steps
                break ## Get out of while loop as we don't want to update the adv-image again
            
            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images, stp_atk
                prev_cost = cost.item()

        return best_adv_images, stp_atk

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

        if self._targeted:
            return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)

class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, steps=100, overshoot=0.02):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']

    def forward(self, images, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        outputs = self.model(images)
        _, labels = torch.max(outputs, dim=1)
        
        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0
        steps_req = [0]*(batch_size)

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre  
                if early_stop:
                    correct[idx] = False
                steps_req[idx] += 1

            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images, steps_req

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)


class APGD(Attack):
    r"""
    APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]
    Distance Measure : Linf, L2
    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: None)
        steps (int): number of steps. (Default: 100)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
        
    Examples::
        >>> attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, norm='Linf', eps=8/255, steps=100, n_restarts=1, 
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False):
        super().__init__("APGD", model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self._supported_mode = ['default']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        self.steps_2, self.steps_min, self.size_decr = max(int(0.22 * self.steps), 1), max(int(0.06 * self.steps), 1), max(int(0.03 * self.steps), 1)
        if self.verbose:
            print('parameters: ', self.steps, self.steps_2, self.steps_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknowkn loss')
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                    
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.steps):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.steps_min)
              
        return x_best, acc, loss_best, x_best_adv
    

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
            
            return acc, adv
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best

class APGDT(Attack):
    r"""
    APGD-Targeted in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.'
    Targeted attack for every wrong classes.
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]
    Distance Measure : Linf, L2
    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: None)
        steps (int): number of steps. (Default: 100)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)
        n_classes (int): number of classes. (Default: 10)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, norm='Linf', eps=8/255, steps=100, n_restarts=1,
                 seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10):
        super().__init__("APGDT", model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.n_target_classes = n_classes - 1
        self._supported_mode = ['default']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images
    
    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)
        
        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)
    
    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        self.steps_2, self.steps_min, self.size_decr = max(int(0.22 * self.steps), 1), max(int(0.06 * self.steps), 1), max(int(0.03 * self.steps), 1)
        if self.verbose:
            print('parameters: ', self.steps, self.steps_2, self.steps_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        output = self.model(x)
        y_target = output.sort(dim=1)[1][:, -self.target_class]
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()
            
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.steps):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.steps_min)
              
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        
        if not cheap:
            raise ValueError('not implemented yet')
        
        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))
                                
        return acc, adv

class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.
    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.
    Examples::
        >>> atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk = torchattacks.MultiAttack([atk1, atk2])
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, attacks, verbose=False):
        
        # Check validity
        ids = []
        print("\nAttacks in AutoAttack")
        for attack in attacks:
            print(attack)
            ids.append(id(attack.model))
        exit(0)
        if len(set(ids)) != 1:
            raise ValueError("At least one of attacks is referencing a different model.")

        super().__init__("MultiAttack", attack.model)
        self.attacks = attacks
        self.verbose = verbose
        self._accumulate_multi_atk_records = False
        self._multi_atk_records = [0.0]
        self._supported_mode = ['default']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            outputs = self.model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = (pre == labels[fails])
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(torch.arange(fails.shape[0]).to(self.device), wrongs)

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

            if len(fails) == 0:
                break

        if self.verbose:
            print(self._return_sr_record(multi_atk_records))

        if self._accumulate_multi_atk_records:
            self._update_multi_atk_records(multi_atk_records)

        return final_images

    def _clear_multi_atk_records(self):
        self._multi_atk_records = [0.0]

    def _covert_to_success_rates(self, multi_atk_records):
        sr = [((1-multi_atk_records[i]/multi_atk_records[0])*100) for i in range(1, len(multi_atk_records))]
        return sr

    def _return_sr_record(self, multi_atk_records):
        sr = self._covert_to_success_rates(multi_atk_records)
        return "Attack success rate: "+" | ".join(["%2.2f %%"%item for item in sr])

    def _update_multi_atk_records(self, multi_atk_records):
        for i, item in enumerate(multi_atk_records):
            self._multi_atk_records[i] += item

    def save(self, data_loader, save_path=None, verbose=True, return_verbose=False, save_pred=False):
        r"""
        Overridden.
        """            
        self._clear_multi_atk_records()
        prev_verbose = self.verbose
        self.verbose = False
        self._accumulate_multi_atk_records = True

        for i, attack in enumerate(self.attacks):
            self._multi_atk_records.append(0.0)

        if return_verbose:
            rob_acc, l2, elapsed_time = super().save(data_loader, save_path, 
                                                     verbose, return_verbose,
                                                     save_pred=save_pred)
            sr = self._covert_to_success_rates(self._multi_atk_records)
        elif verbose:
            super().save(data_loader, save_path, verbose,
                         return_verbose, save_pred=save_pred)
            sr = self._covert_to_success_rates(self._multi_atk_records)
        else:
            super().save(data_loader, save_path, verbose=False,
                         return_verbose=False, save_pred=save_pred)
            
        self._clear_multi_atk_records()
        self._accumulate_multi_atk_records = False
        self.verbose = prev_verbose

        if return_verbose:
            return rob_acc, sr, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        r"""
        Overridden.
        """
        print("- Save progress: %2.2f %% / Robust accuracy: %2.2f %%"%(progress, rob_acc)+\
              " / "+self._return_sr_record(self._multi_atk_records)+\
              ' / L2: %1.5f (%2.3f it/s) \t'%(l2, elapsed_time), end=end)

class AutoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self._supported_mode = ['default']

        if version == 'standard':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
            ])

        elif version == 'plus':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=5),
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='dlr', n_restarts=5),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=5),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, targeted=True, n_classes=n_classes, n_restarts=1),
            ])

        elif version == 'rand':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', eot_iter=20, n_restarts=1),
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='dlr', eot_iter=20, n_restarts=1),
            ])

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed