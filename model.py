import torch
from torch import nn, einsum, Tensor
from einops import rearrange, repeat
import copy
from typing import Optional, List
import torch.nn.functional as F
import math

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def position_code(row, cow):
    pe = torch.zeros(row, cow)
    position = torch.arange(0., row).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., cow, 2) * -
                         1 * (math.log(10000.0) / cow))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)[:, 0:cow // 2]
    return pe


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)        
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
       
        output = src.permute(1, 0, 2)

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos.permute(1,0,2)

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

               
class KernelAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads > 0 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, kx, krd, clst, att_mask=None, l_debug_idx=0):
        c_qkv = self.to_qkv(x).chunk(3, dim = -1)
        k_kqv = self.to_qkv(kx).chunk(3, dim = -1)
        c_kqv = self.to_qkv(clst).chunk(3, dim = -1)

        t_q, t_k, t_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), c_qkv)
        k_q, k_k, k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), k_kqv)
        c_q, _  , _   = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), c_kqv)

        dots = einsum('b h i d, b h j d -> b h i j', t_q, k_k) * self.scale
        if att_mask is not None:
            dots = dots.masked_fill(att_mask, torch.tensor(-1e9))
        attn_ = self.attend(dots*24)* krd.permute(0,1,3,2)
        attn = self.attend(dots)* krd.permute(0,1,3,2)
        att_out = einsum('b h i j, b h j d -> b h i d', attn, k_v)
        att_out = rearrange(att_out, 'b h n d -> b n (h d)')

        k_dots = einsum('b h i d, b h j d -> b h i j', k_q, t_k) * self.scale
        if att_mask is not None:
            k_dots = k_dots.masked_fill(att_mask.permute(0,1,3,2), torch.tensor(-1e9))
        k_attn = self.attend(k_dots) * krd
        k_out = einsum('b h i j, b h j d -> b h i d', k_attn, t_v)
        k_out = rearrange(k_out, 'b h n d -> b n (h d)')

        c_dots = einsum('b h i d, b h j d -> b h i j', c_q, k_k) * self.scale
        if att_mask is not None:
            c_dots = c_dots.masked_fill(att_mask[:,:,:1], torch.tensor(-1e9))
        c_attn = self.attend(c_dots)
        c_out = einsum('b h i j, b h j d -> b h i d', c_attn, k_v)
        c_out = rearrange(c_out, 'b h n d -> b n (h d)')

        return self.to_out(att_out), self.to_out(k_out), self.to_out(c_out), attn_.permute(0,1,3,2)


class KATBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                KernelAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
        self.h = heads
        self.dim = dim

    def forward(self, x, kx, rd, clst, mask=None, kmask=None):
        kernel_mask = repeat(kmask, 'b i ()  -> b i j', j = self.dim) < 0.5
        att_mask = einsum('b i d, b j d -> b i j', mask.float(), kmask.float())
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h = self.h) < 0.5

        rd = repeat(rd.unsqueeze(1), 'b () i j -> b h i j', h = self.h)
        soft_mask = rd

        k_reps = []
        atten_map = []
        for l_idx, (pn, attn, ff) in enumerate(self.layers):
            x, kx, clst = pn(x), pn(kx), pn(clst)

            x_, kx_, clst_, atten_map_ = attn(x, kx, soft_mask, clst, att_mask, l_idx)
            x = x + x_
            clst = clst + clst_
            kx = kx + kx_
            
            x = ff(x) + x
            clst = ff(clst) + clst
            kx = ff(kx) + kx

            k_reps.append(kx.masked_fill(kernel_mask, 0))
            atten_map.append(atten_map_)

        return k_reps, clst, atten_map


class FGCR(nn.Module):
    def __init__(self, patch_dim, prompt_num, dim, depth, heads, mlp_dim, prompt_list, vocab_size=300, t_head=4, t_n_layer=6, t_d_model=256, t_d_ff=512, 
                 t_dropout=0.5, num_kernel=25, dim_head = 64, dropout = 0.5, emb_dropout = 0.,emd_dim= 128, max_position_embeddings=175):
        super().__init__()

        # Image_encoder
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.img_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, dim))
        self.nk = num_kernel
        self.kt = KATBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Text Encoder
        self.prompt = prompt_list
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.text_position_embeddings = nn.Embedding(max_position_embeddings, dim)
        self.text_embedding = nn.Embedding(vocab_size, emd_dim)
        self.text_linear = nn.Linear(emd_dim, dim)
        text_encoder_layer = TransformerEncoderLayer(t_d_model, t_head, t_d_ff,
                                        t_dropout)
        text_encoder_norm = nn.LayerNorm(t_d_model) 
        self.text_encoder = TransformerEncoder(text_encoder_layer, t_n_layer, text_encoder_norm)
        self.LayerNorm = nn.LayerNorm(t_d_model) 
        self.text_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.text_head = BertLMPredictionHead(dim, vocab_size)

        # Local sructure
        self.local_atten_layer = nn.MultiheadAttention(
            dim, heads)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, prompt_num) 
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        self.activate = nn.Tanh()

    def forward(self, node_features, krd, text, mask=None, kmask=None, tmask=None, pmask=None):
        # Extract image features
        x = self.to_patch_embedding(node_features)
        b = x.shape[0]
        cls_tokens = repeat(self.img_cls_token, '() n d -> b n d', b = b)
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b = b, k = self.nk)
        x = self.dropout(x)
        k_reps, clst, _ = self.kt(x, kernel_tokens, krd, cls_tokens, mask, kmask)
        anchor_ebd = k_reps[5]
        img_cls = clst[:,0]
        img_cls = self.activate(img_cls)

        # Extract text features
        t = self.text_embedding(text)
        t = self.text_linear(t)
        position_ids = self.position_ids[:, 0 : t.size()[1]]
        position_embeddings = self.text_position_embeddings(position_ids)
        t = t.squeeze(dim=2)
        t += position_embeddings
        text_cls_tokens = repeat(self.text_cls_token, '() n d -> b n d', b = b)
        t = torch.cat((text_cls_tokens, t), dim=1)
        t = self.LayerNorm(t)
        t_feat = self.dropout(t)        
        tmp = torch.ones((tmask.size()[0],1,tmask.size()[2])).int().cuda(non_blocking=True)    
        t_inmask = torch.cat((tmp, tmask), dim=1)[:,:,0]
        t_inmask = t_inmask<0.5
        t_out = self.text_encoder(t_feat, src_key_padding_mask=t_inmask)
        text_cls = t_out[:,0]
        text_ebd = t_out[:,1:]
        text_cls = self.activate(text_cls)

        # Extract prompt features
        prompt_index = torch.tensor(self.prompt).int().cuda(non_blocking=True).long()
        prompt_index = prompt_index.repeat(b,1,1).permute(0, 2, 1)
        p = self.text_embedding(prompt_index)
        p = self.text_linear(p)
        position_ids = self.position_ids[:, 0 : prompt_index.size()[1]]
        position_embeddings = self.text_position_embeddings(position_ids)
        p = p.squeeze(dim=2)
        p += position_embeddings        
        p = self.LayerNorm(p)
        p = self.dropout(p) 
        prompt_ebd = self.text_encoder(p)

        return img_cls, anchor_ebd, text_cls, t_feat, text_ebd, prompt_ebd

    def loss(self, img_cls, anchor_ebd, text_cls, t_feat, text_ebd, prompt_ebd, prompt, text, tmask,  kmask, pmask):
        # Masked Language Modeling (MLM) 
        t_rnd_mask = torch.rand(tmask.size(), out=None).type_as(img_cls)<0.6
        tmp = torch.ones((tmask.size()[0],1,tmask.size()[2])).int().cuda(non_blocking=True)    
        t_inmask_hide = torch.cat((tmp, t_rnd_mask*tmask), dim=1)[:,:,0]
        t_inmask_hide = t_inmask_hide<0.5
        t_hide = self.text_encoder(t_feat,src_key_padding_mask=t_inmask_hide)
        text_ebd_hide = t_hide[:,1:]
        mlm_logits = self.text_head(text_ebd_hide)
        mlm_logits = torch.sigmoid(mlm_logits)
        mlm_loss = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1))[tmask.view(-1)>0], text.view(-1)[tmask.view(-1)>0])*0.1

        # Prompt Classification (PC) -- Eq.11
        text_cls = self.mlp_head(text_cls)
        img_cls = self.mlp_head(img_cls)      
        pc_loss =  multi_cls_loss(img_cls, prompt, pmask) + multi_cls_loss(text_cls, prompt, pmask)

        # Anchor-Prompt Alignment (APA) -- Eq.7
        apa_loss, soft_pred, = APA_Loss(anchor_ebd, kmask, prompt_ebd, prompt, pmask)

        # Coss-attention Token Alignment (CTA) -- Eq.10
        t_inmask = torch.cat((tmp, tmask), dim=1)[:,:,0]
        t_inmask = t_inmask<0.5
        patch_atten_output, _ = self.local_atten_layer(
            anchor_ebd.permute(1, 0, 2), text_ebd.permute(1, 0, 2), text_ebd.permute(1, 0, 2), key_padding_mask=t_inmask[:,1:])
        text_atten_output, _ = self.local_atten_layer(
            text_ebd.permute(1, 0, 2), anchor_ebd.permute(1, 0, 2), anchor_ebd.permute(1, 0, 2), key_padding_mask=kmask[:,:,0]<0.5)
        loss_patch = cross_sim_loss(anchor_ebd, patch_atten_output, kmask)
        loss_text = cross_sim_loss(text_ebd, text_atten_output, tmask)
        cta_loss = loss_text+loss_patch

        # WSI-Report Alignment (WRA) -- Eq.6
        wra_loss = WRA_Loss(img_cls, text_cls)

        loss = apa_loss+wra_loss+mlm_loss+cta_loss+pc_loss
        return loss

def multi_cls_loss(im_out, prompt, pmask):
    target = torch.zeros(im_out.size()).cuda()
    ce_loss = torch.zeros(im_out.size()[0]).cuda()
    for i in range(im_out.size()[0]):
        target[i,prompt[i][pmask[i]>0]]=1 
        ce_loss[i] = F.binary_cross_entropy(torch.softmax(im_out[i].clone(),dim=0), target[i].clone())

    return ce_loss.mean()

def APA_Loss(img_feat, kmask, p_ebd, prompt, pmask, T=10):
    img_text_matrix =  einsum('b i j, b j d -> b i d', img_feat, p_ebd.permute(0, 2, 1))/img_feat.size()[2]/T
    target = torch.zeros(p_ebd.size()[0:2]).cuda()
    ce_loss = torch.zeros(p_ebd.size()[0]).cuda()
    soft_pred = torch.zeros(p_ebd.size()[0:2]).cuda()
    soft_pred_k = torch.zeros(img_text_matrix.size()).cuda()
    for i in range(img_text_matrix.size()[0]):
        target[i,prompt[i][pmask[i]>0]]=1 
        kmask_part = kmask[i].reshape(img_text_matrix.size()[1])>0
        img_text_matrix_part = img_text_matrix[i][kmask_part]
        pred_logit_part = torch.sigmoid(img_text_matrix_part)
        soft_pred_part = torch.softmax(img_text_matrix_part,dim=0) *pred_logit_part
        soft_pred_k[i][kmask_part] = soft_pred_part
        soft_pred[i] = soft_pred_part.sum(0)
        ce_loss[i] = F.binary_cross_entropy(soft_pred[i].clone(), target[i].clone())

    return ce_loss.mean(), soft_pred

def WRA_Loss(img_rep, text_rep):
    bz = img_rep.size(0)
    labels = torch.arange(bz).type_as(text_rep).long()
    scores = img_rep.mm(text_rep.t())/img_rep.size(1)
    scores1 = scores.transpose(0, 1)
    loss0 = F.cross_entropy(scores, labels)
    loss1 = F.cross_entropy(scores1, labels)
    loss_ita = (loss0 + loss1)*0.2

    return loss_ita

def cross_sim_loss(anchor_ebd, data_atten_output, mask):
    data_sim = torch.bmm(anchor_ebd, data_atten_output.permute(
        1, 2, 0)) / anchor_ebd.size(2)
    data_num = data_sim.size(1)
    bz = data_sim.size(0)
    mask = rearrange(mask, "b n1 n2 -> (b n1 n2)")>0
    data_sim_1 = rearrange(data_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(data_num).type_as(
        data_sim).long().repeat(bz)
    loss_data_1 = torch.sum(F.cross_entropy(
        data_sim_1[mask], targets[mask], reduction="none") ) / mask.sum()

    data_sim_2 = rearrange(data_sim, "b n1 n2 -> (b n2) n1")
    loss_data_2 = torch.sum(F.cross_entropy(
        data_sim_2[mask], targets[mask], reduction="none") ) / mask.sum()
    loss = (loss_data_1 + loss_data_2) * 0.01

    return loss

def unpack_data(data):
    feats = data[0].float().cuda(non_blocking=True)
    rd = data[1].float().cuda(non_blocking=True)
    text = data[2].int().cuda(non_blocking=True).long()
    prompt = data[3].int().cuda(non_blocking=True).long()
    mask = data[4].int().cuda(non_blocking=True)
    kmask = data[5].int().cuda(non_blocking=True)
    tmask = data[6].int().cuda(non_blocking=True)
    pmask = data[7].int().cuda(non_blocking=True)
    wsi_label = data[10].int().cuda(non_blocking=True).long()

    return feats, rd, text, prompt, mask, kmask, tmask, pmask, wsi_label
