import torch
import torch.nn.functional as F


def SFA(x, y, model, resize_factor=1., x_a=None, targeted=False, max_queries=1000, linf=0.031):
    '''
    Sign Flip Attack: linf decision-based adversarial attack
    :param x: original images, torch tensor of size (b,c,h,w)
    :param y: original labels for untargeted attacks, target labels for targeted attacks, torch tensor of size (b,)
    :param model: target model
    :param resize_factor: dimensionality reduction rate, > 1.0
    :param x_a: initial images for targeted attacks, torch tensor of size (b,c,h,w). None for untargeted attacks
    :param targeted: attack mode, True for targeted attacks, False for untargeted attacks
    :param max_queries: maximum query number
    :param linf: linf threshold
    :return: adversarial examples and corresponding required queries
    '''
    
    # initialize
    if targeted:
        assert x_a is not None
        check = is_adversarial(x_a, y, model, targeted)
        if check.sum() < y.size(0):
            print('Some initial images do not belong to the target class!')
            return x, torch.zeros(x.size(0))
        check = is_adversarial(x, y, model, targeted)
        if check.sum() > 0:
            print('Some original images already belong to the target class!')
            return x, torch.zeros(x.size(0))
    else:
        check = is_adversarial(x, y, model, True)
        if check.sum() < y.size(0):
            print('Some original images do not belong to the original class!')
            return x, torch.zeros(x.size(0))
        x_a = torch.rand_like(x)
        iters = 0
        check = is_adversarial(x_a, y, model, targeted)
        while check.sum() < y.size(0):
            x_a[check < 1] = torch.rand_like(x_a[check < 1])
            check = is_adversarial(x_a, y, model, targeted)
            iters += 1
            if iters > 10000:
                print('Initialization Failed!')
                return
    # linf binary search
    x_a = binary_infinity(x_a, x, y, 10, model, targeted)
    delta = x_a - x
    del x_a
    
    b, c, h, w = delta.size()

    assert resize_factor >= 1.
    h_dr, w_dr = int(h // resize_factor), int(w // resize_factor)

    # Q: query number for each image
    Q = torch.zeros(b)
    # q_num: current queries
    q_num = 0
    # 10 queries for binary search
    q_num, Q= q_num + 10, Q + 10

    # indices for unsuccessful images
    unsuccessful_indices = torch.ones(b) > 0

    # hyper-parameter initialization
    alpha = torch.ones(b) * 0.004
    prob = torch.ones_like(delta) * 0.999
    prob = resize(prob, h_dr, w_dr)

    # additional counters for hyper-parameter adjustment
    reset = 0
    proj_success_rate = torch.zeros(b)
    flip_success_rate = torch.zeros(b)
    
    while q_num < max_queries:
        reset += 1
        b_cur = unsuccessful_indices.sum()

        # the project step
        eta = torch.randn([b_cur,c,h_dr,w_dr]).sign() * alpha[unsuccessful_indices][:,None,None,None]
        eta = resize(eta, h, w)
        l, _ = delta[unsuccessful_indices].abs().view(b_cur, -1).max(1)
        delta_p = project_infinity(delta[unsuccessful_indices]+eta, torch.zeros_like(eta), l-alpha[unsuccessful_indices])
        check = is_adversarial((x[unsuccessful_indices] + delta_p).clamp(0,1), y[unsuccessful_indices], model, targeted)
        delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_p[check.nonzero().squeeze(1)]
        proj_success_rate[unsuccessful_indices] += check.float()

        # the random sign flip step
        s = torch.bernoulli(prob[unsuccessful_indices]) * 2 - 1
        delta_s = delta[unsuccessful_indices] * resize(s, h, w).sign()
        check = is_adversarial((x[unsuccessful_indices] + delta_s).clamp(0, 1), y[unsuccessful_indices], model,
                               targeted)
        prob[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] -= s[check.nonzero().squeeze(
            1)] * 1e-4
        prob.clamp_(0.99, 0.9999)
        flip_success_rate[unsuccessful_indices] += check.float()
        delta[unsuccessful_indices.nonzero().squeeze(1)[check.nonzero().squeeze(1)]] = delta_s[
            check.nonzero().squeeze(1)]

        # hyper-parameter adjustment
        if reset % 10 == 0:
            proj_success_rate /= reset
            flip_success_rate /= reset
            alpha[proj_success_rate > 0.7] *= 1.5
            alpha[proj_success_rate < 0.3] /= 1.5
            prob[flip_success_rate > 0.7] -= 0.001
            prob[flip_success_rate < 0.3] += 0.001
            prob.clamp_(0.99, 0.9999)
            reset = 0
            proj_success_rate *= 0
            flip_success_rate *= 0

        # query count
        q_num += 2
        Q[unsuccessful_indices] += 2

        # update indices for unsuccessful perturbations
        l, _ = delta[unsuccessful_indices].abs().view(b_cur, -1).max(1)
        unsuccessful_indices[unsuccessful_indices.nonzero().squeeze(1)[(l <= linf).nonzero().squeeze(1)]] = 0

        # print attack information
        if q_num % 10000 == 0:
            print(f"Queries: {q_num}/{max_queries} Successfully attacked images: {b - unsuccessful_indices.sum()}/{b}")

        if unsuccessful_indices.sum() == 0:
            break

    print('attack finished!')
    print(f"Queries: {q_num}/{max_queries} Successfully attacked images: {b - unsuccessful_indices.sum()}/{b}")
    return (x+delta).clamp(0,1), Q

def resize(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bilinear', align_corners=False)

def binary_infinity(x_a, x, y, k, model, targeted):
    '''
    linf binary search
    :param k: the number of binary search iteration
    '''
    size = x_a.size()
    l = torch.zeros(size[0]).view(size[0], 1, 1, 1)
    u, _ = (x_a - x).view(size[0], -1).abs().max(1)
    u = u.view(size[0], 1, 1, 1)
    for _ in range(k):
        mid = (l + u) / 2
        adv = torch.max(x - mid, torch.min(x_a, x + mid)).clamp(0, 1)
        still = is_adversarial(adv, y, model, targeted)
        u[still.nonzero().squeeze(1)] = mid[still.nonzero().squeeze(1)]
        still = still < 1
        l[still.nonzero().squeeze(1)] = mid[still.nonzero().squeeze(1)]
    return torch.max(x - u, torch.min(x_a, x + u)).clamp(0, 1)

def project_infinity(x_a, x, l):
    '''
    linf projection
    '''
    return torch.max(x - l[:, None, None, None], torch.min(x_a, x + l[:, None, None, None]))

def get_predict_label(x, model):
    return model(normalize(x).cuda()).cpu().argmax(1)

def normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return (x - torch.tensor(mean)[None,:,None,None]) / torch.tensor(std)[None,:,None,None]

def is_adversarial(x, y, model, targeted=False):
    '''
    check whether the adversarial constrain holds for x
    '''
    if targeted:
        return get_predict_label(x, model) == y
    else:
        return get_predict_label(x, model) != y
