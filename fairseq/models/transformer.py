import math
import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import LearnedPositionalEmbedding, GradMultiply, LinearizedConvolution, LayerNormalization, BeamableMM
from fairseq.data import LanguagePairDataset

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel

class TransformerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""
    def __init__(self, dictionary, embed_dim=256, max_positions=1024,
                 num_layers=2, num_heads=8,
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        super().__init__(dictionary)
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx,
                                                   left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)

        self.layers = num_layers
        
        self.self_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttention(hidden_size, hidden_size, hidden_size, num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
            self.norm1_blocks.append(LayerNormalization(hidden_size))
            self.norm2_blocks.append(LayerNormalization(hidden_size))
        self.out_norm = LayerNormalization(hidden_size)

    def forward(self, src_tokens):
        # embed tokens plus positions
        input_to_padding = attention_bias_ignore_padding(src_tokens, self.dictionary.pad())
        encoder_self_attention_bias = encoder_attention_bias(input_to_padding)
        encoder_input = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(encoder_input, p=self.dropout, training=self.training)
        
        for self_attention, ffn, norm1, norm2 in zip(self.self_attention_blocks, 
                                                     self.ffn_blocks,
                                                     self.norm1_blocks,
                                                     self.norm2_blocks):
            y = self_attention(norm1(x), None, encoder_self_attention_bias)
            x = residual(x, y, self.dropout, self.training)
            y = ffn(norm2(x))
            x = residual(x, y, self.dropout, self.training)
        x = self.out_norm(x)
        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()
        

class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module('bmm', BeamableMM(beamable_mm_beam_size))



class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""
    def __init__(self, dictionary, embed_dim=256, max_positions=1024,
                 num_layers=2, num_heads=8,
                 filter_size=256, hidden_size=256,
                 dropout=0.1, attention_dropout=0.1, relu_dropout=0.1, share_embed=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx,
                                                   left_pad=LanguagePairDataset.LEFT_PAD_TARGET)

        self.layers = num_layers

        self.self_attention_blocks = nn.ModuleList()
        self.encdec_attention_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.norm1_blocks = nn.ModuleList()
        self.norm2_blocks = nn.ModuleList()
        self.norm3_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.self_attention_blocks.append(MultiheadAttentionDecoder(hidden_size, hidden_size, hidden_size, num_heads))
            self.ffn_blocks.append(FeedForwardNetwork(hidden_size, filter_size, relu_dropout))
            self.norm1_blocks.append(LayerNormalization(hidden_size))
            self.norm2_blocks.append(LayerNormalization(hidden_size))
            self.norm3_blocks.append(LayerNormalization(hidden_size))
            self.encdec_attention_blocks.append(MultiheadAttention(hidden_size, hidden_size, hidden_size, num_heads, to_weights=True))

        if share_embed:
            assert out_embed_dim == embed_dim, \
                "Shared embed weights implies same dimensions " \
                " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
            self.out_embed = nn.Linear(hidden_size, num_embeddings)
            self.out_embed.weight = self.embed_tokens.weight
        else:
            self.out_embed = Linear(hidden_size, num_embeddings, dropout=dropout)

    def forward(self, input_tokens, encoder_out):
        # split and transpose encoder outputs

        input_to_padding = attention_bias_ignore_padding(input_tokens, self.dictionary.pad())
        decoder_self_attention_bias = encoder_attention_bias(input_to_padding)
        decoder_self_attention_bias += attention_bias_lower_triangle(input_tokens)
        # embed positions
        positions = self.embed_positions(input_tokens)
        
        if self._is_incremental_eval:
            input_tokens = input_tokens[:, -1:]
            decoder_self_attention_bias = decoder_self_attention_bias[:, -1:, :]
        # embed tokens and positions
        x = self.embed_tokens(input_tokens) + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x
        
        avg_attn_scores = None
        num_attn_layers = len(self.encdec_attention_blocks)
        for self_attention, encdec_attention, ffn, norm1, norm2, norm3 in zip(self.self_attention_blocks,
                                                                              self.encdec_attention_blocks,
                                                                              self.ffn_blocks,
                                                                              self.norm1_blocks,
                                                                              self.norm2_blocks,
                                                                              self.norm3_blocks):
            y = self_attention(norm1(x), None, decoder_self_attention_bias)
            x = residual(x, y, self.dropout, self.training)
            
            y, attn_scores = encdec_attention(norm2(x), encoder_out, None)
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)
            x = residual(x, y, self.dropout, self.training)

            y = ffn(norm3(x))
            x = residual(x, y, self.dropout, self.training)
        avg_attn_scores = avg_attn_scores / self.layers
        x = self.out_embed(x)
        return x, avg_attn_scores

    def reorder_incremental_state(self, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        super().reorder_incremental_state(new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict


class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, 
                 key_depth, value_depth, output_depth,
                 num_heads, dropout=0.1, to_weights=False):
        super(MultiheadAttention, self).__init__()

        self._query = Linear(key_depth, key_depth, bias=False)
        self._key = Linear(key_depth, key_depth, bias=False)
        self._value = Linear(value_depth, value_depth, bias=False)
        self.output_perform = Linear(value_depth, output_depth, bias=False)

        self.num_heads = num_heads
        self.key_depth_per_head = key_depth // num_heads
        self.dropout = dropout
        self.to_weights = to_weights
        
    def forward(self, query_antecedent, memory_antecedent, bias):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        q = self._query(query_antecedent)
        k = self._key(memory_antecedent)
        v = self._value(memory_antecedent)
        q *= self.key_depth_per_head ** -0.5
        
        # split heads
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        x = []
        avg_attn_scores = None
        for i in range(self.num_heads):
            results = dot_product_attention(q[i], k[i], v[i],
                                            bias,
                                            self.dropout, self.to_weights)
            if self.to_weights:
                y, attn_scores = results
                if avg_attn_scores is None:
                    avg_attn_scores = attn_scores
                else:
                    avg_attn_scores.add_(attn_scores)
            else:
                y = results
            x.append(y)
        x = combine_heads(x)
        x = self.output_perform(x)
        if self.to_weights:
            return x, avg_attn_scores / self.num_heads
        else:
            return x


class MultiheadAttentionDecoder(MultiheadAttention):
    def __init__(self,
                 key_depth, value_depth, output_depth,
                 num_heads, dropout=0.1):
        super(MultiheadAttentionDecoder, self).__init__(key_depth, value_depth, output_depth,
                                                        num_heads, dropout)
        self._is_incremental_eval = False

    def incremental_eval(self, mode=True):
        self._is_incremental_eval = mode
        if mode:
            self.clear_incremental_state()

    def forward(self, query_antecedent, memory_antecedent, bias):
        if self._is_incremental_eval:
            return self.incremental_forward(query_antecedent, memory_antecedent, bias)
        else:
            return super().forward(query_antecedent, memory_antecedent, bias)
    
    def incremental_forward(self, query_antecedent, memory_antecedent, bias):
        if memory_antecedent is None:
            memory_antecedent = query_antecedent
        q = self._query(query_antecedent)
        k = self._key(memory_antecedent)
        v = self._value(memory_antecedent)
        q *= self.key_depth_per_head ** -0.5
        if self.key_buffer is None:
            self.key_buffer = k.clone()
            self.value_buffer = v.clone()
        else:
            self.key_buffer = torch.cat([self.key_buffer, k], 1)
            self.value_buffer = torch.cat([self.value_buffer, v], 1)

        k = self.key_buffer
        v = self.value_buffer
        
        # split heads
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        x = []
        for i in range(self.num_heads):
            x.append(dot_product_attention(q[i], k[i], v[i], bias, self.dropout))
        x = combine_heads(x)
        x = self.output_perform(x)
        return x

    def clear_incremental_state(self):
        """
        Key Buffer: [Batch size, Length, Dimmension]
        """
        self.key_buffer = None
        self.value_buffer = None

    def reorder_incremental_state(self, new_order):
        if self.key_buffer is not None:
            self.key_buffer.data = self.key_buffer.data.index_select(0, new_order)
            self.value_buffer.data = self.value_buffer.data.index_select(0, new_order)



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = Linear(hidden_size, filter_size, bias=False)
        self.fc2 = Linear(filter_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def residual(x, y, dropout, training):
    y = F.dropout(y, p=dropout, training=training)
    return x + y


def split_heads(x, num_heads):
    """split x into multi heads
    Args:
        x: [batch_size, length, depth]
    Returns:
        y: [[batch_size, length, depth / num_heads] x heads]
    """
    sz = x.size()
    # x -> [batch_size, length, heads, depth / num_heads]
    x = x.view(sz[0], sz[1], num_heads, sz[2] // num_heads)
    # [batch_size, length, 1, depth // num_heads] * 
    heads = torch.chunk(x, num_heads, 2)
    x = []
    for i in range(num_heads):
        x.append(torch.squeeze(heads[i], 2))
    return x


def combine_heads(x):
    """combine multi heads
    Args:
        x: [batch_size, length, depth / num_heads] x heads
    Returns:
        x: [batch_size, length, depth]
    """
    return torch.cat(x, 2)
    

def dot_product_attention(q, k, v, bias, dropout, to_weights=False):
    """dot product for query-key-value
    Args:
        q: query antecedent, [batch, length, depth]
        k: key antecedent,   [batch, length, depth]
        v: value antecedent, [batch, length, depth]
    """
    # [batch, length, depth] x [batch, depth, length] -> [batch, length, length]
    logits = torch.bmm(q, k.transpose(1, 2).contiguous())
    if bias is not None:
        logits += bias
    size = logits.size()
    weights = F.softmax(logits.view(size[0] * size[1], size[2]), dim=1)
    weights = weights.view(size)
    if to_weights:
        return torch.bmm(weights, v), weights
    else:
        return torch.bmm(weights, v)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


def attention_bias_ignore_padding(src_tokens, padding_idx):
    """Calculate the padding mask based on which embedding are zero
    Args:
        src_tokens: [batch_size, length]
    Returns:
        bias: [batch_size, length]
    """
    return src_tokens.eq(padding_idx).unsqueeze(1)


def attention_bias_lower_triangle(input_tokens):
    batch_size, length = input_tokens.size()
    attention_bias_lower_triangle = torch.triu(input_tokens.data.new(length, length).fill_(1), diagonal=1)
    return Variable(attention_bias_lower_triangle.expand(batch_size, length, length).float()) * -1e9


def encoder_attention_bias(bias):
    batch_size, _, length = bias.size()
    return bias.expand(batch_size, length, length).float() * -1e9


def get_archs():
    return ['transformer_iwslt_de_en', 'transformer', 'transformer_wmt_en_de', 'transformer_lmc_zh_en']


def _check_arch(args):
    pass


def parse_arch(args):
    _check_arch(args)

    if args.arch == 'transformer_iwslt_de_en':
        args.hidden_size = 256
        args.filter_size = 1024
        args.num_heads = 4
        args.num_layers = 2

        args.encoder_embed_dim = 256
        args.encoder_layers = '[(256, 3)] * 4'
        args.decoder_embed_dim = 256
        args.decoder_layers = '[(256, 3)] * 3'
        args.decoder_out_embed_dim = 256
    else:
        assert args.arch == 'transformer'

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)

    return args


def build_model(args, src_dict, dst_dict):
    encoder = TransformerEncoder(
        src_dict,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        filter_size=args.filter_size
    )
    decoder = TransformerDecoder(
        dst_dict,
    )
    return TransformerModel(encoder, decoder)

