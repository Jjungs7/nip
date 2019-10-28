import torch, torch.nn as nn, torch.nn.functional as F
from layers import *

import os, random, math, numpy as np
import logging, colorlog
logging.disable(logging.DEBUG)
colorlog.basicConfig(
    filename=None,
    level=logging.NOTSET,
    format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
class Classifier(nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()

        self.word_em = BasicWordEmb(args)
        self.encoder = BasicBiLSTM(args)
        self.attention = BasicAttention(args)

        if args.model_type == 'linear_basis_cust':
            self.W = BasisCustLinear(args)
        elif args.model_type == 'stochastic_linear_basis_cust':
            self.W = StochasticBasisCustLinear(args)
        else:
            self.W = BasicLinear(args)

        self.b = BasicBias(args)


    def init_param(self, save_init_param=False):
        # Manual Random Seed
        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed(self.args.random_seed)
        torch.backends.cudnn.deterministic=True

        self.word_em.init_param()
        self.encoder.init_param()
        self.attention.init_param()
        self.W.init_param()
        self.b.init_param()        
        if self.args.pretrained_word_em_dir:
            colorlog.info("[init_param] for word emb: from {}".format(self.args.pretrained_word_em_dir))
            word_em = np.load(self.args.pretrained_word_em_dir)
            self.word_em.word_em.weight.data.copy_(torch.from_numpy(word_em))
        if save_init_param:
            fname = "Seed{}.init_param".format(self.args.random_seed)+'.pth'
            torch.save(self.state_dict(), os.path.join(self.args.param_dir, fname))
    def forward(self, review, length, mask, **kwargs):
        x = self.word_em(review, **kwargs)
        # 2. BiLSTM 
        x = self.encoder(x, length, **kwargs)
        # 3. Attention
        x = self.attention(x, mask, **kwargs)
        # 4. FC Weight Matrix
        x = self.W(x, **kwargs)
        # 5. FC bias
        b = self.b(**kwargs)
        while b.dim()<x.dim(): b.unsqueeze_(1)
        x += b
        return x
class StochasticClassifier(Classifier):
    def __init__(self, args):
        super().__init__(args)
    def forward(self, review, length, mask, **kwargs):
        output = super().forward(review, length, mask, **kwargs)
        probs = F.softmax(output, -1).mean(1)
        if self.training:
            return probs, output
        else:
            return probs
    def get_loss(self, output, label):
        gather = output[torch.arange(output.shape[0]), label]
        return (-torch.log(gather)).mean() # Loss from Ground-truth Labels
    def mean_and_variance(self, review, length, mask, **kwargs):
        x = self.word_em(review, **kwargs)
        # 2. BiLSTM 
        x = self.encoder(x, length, **kwargs)
        # 3. Attention
        x = self.attention(x, mask, **kwargs)
        # 4. FC Weight Matrix
        logits_sampled, coef_sampled = self.W.mean_and_variance(x, **kwargs) # (N, S, C), (N, S, B)
        coef_mu, coef_std = coef_sampled.mean(1), coef_sampled.std(1)
        # 5. FC bias
        logits_sampled += self.b(**kwargs).unsqueeze(1) # N, S, C
        logits_mu, logits_std = logits_sampled.mean(1), logits_sampled.std(1)
        
        prob_sampled = F.softmax(logits_sampled, -1)
        prob_mu, prob_std = prob_sampled.mean(1), prob_sampled.std(1)
        return prob_mu, prob_std, logits_mu, logits_std, coef_mu, coef_std
class DeterministicClassifier(Classifier):
    def __init__(self, args):
        super().__init__(args)
    def get_loss(self, output, label):
        return F.cross_entropy(output, label)
    def extract_logits(self, dataloader):
        if not os.path.exists(self.args.extract_logits_result_dir):
            os.mkdir(self.args.extract_logits_result_dir)
        N = len(dataloader.dataset)
        B = dataloader.batch_size
        if 'basis' in self.args.model_type:
            coef_logits_np = np.zeros((N, self.args.num_bases), dtype=np.float32)
        logits_np = np.zeros((N, self.args.num_labels), dtype=np.float32)
        pred_np = np.zeros((N), dtype=np.int64)
        target_np = np.zeros((N), dtype=np.int64)
        with torch.no_grad():
            self.eval()
            for i_sample, sample_batch in enumerate(dataloader):
                (text, length, mask,
                user, product,
                label) = sample_batch
                logits = self(text, length, mask, **{'user':user, 'product':product}) # N, 5
                pred = torch.argmax(logits, dim=-1)

                meta_inputs = {'user':user, 'product':product}
                if 'linear_basis_cust'==self.args.model_type:
                    coef_logits_np[i_sample*B:(i_sample+1)*B] = self.W.C(torch.cat([getattr(self.W, name)(idx) for name, idx in meta_inputs.items()], dim=1)).cpu().data.numpy()

                logits_np[i_sample*B:(i_sample+1)*B] = logits.cpu().data.numpy()
                pred_np[i_sample*B:(i_sample+1)*B] = pred.cpu().data.numpy()
                target_np[i_sample*B:(i_sample+1)*B] = label.cpu().data.numpy()
            self.train()
        acc = (pred_np==target_np).mean()
        rmse = ((pred_np-target_np)**2).mean()**0.5

        dtype = dataloader.dataset.name
        colorlog.info("Performance for {} Dataset >> Accuracy: {:2.1f}%,  RMSE: {:.3f}".format(dtype, acc*100, rmse))
        
        result_path = os.path.join(self.args.extract_logits_result_dir,  "logits_{}.npy".format(dtype))
        np.save(result_path, logits_np)
        colorlog.info("Save extracted logits: shape {}, path {}".format(logits_np.shape, result_path))
        
        if 'basis' in self.args.model_type:
            result_path = os.path.join(self.args.extract_logits_result_dir,  "coef_logits_{}.npy".format(dtype))
            np.save(result_path, coef_logits_np)
            colorlog.info("Save extracted logits for coefficients of bases: shape {}, path {}".format(coef_logits_np.shape, result_path))            
        return acc, rmse