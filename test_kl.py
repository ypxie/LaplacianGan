import torch

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    import pdb; pdb.set_trace()
    KLD = torch.mean(KLD_element).mul_(-0.5)
    
    return KLD

def reparameterize(self, mu, logvar):
    if self.training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu

mu = torch.randn(4,10)
mu[mu < 0] = 0
logvar = torch.randn(4,10)
logvar[logvar < 0] = 0

print (KL_loss(mu, logvar))
# print (KL_loss(mu.mean(0), logvar.mean(0)))

