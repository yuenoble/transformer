@[TOC](Transformer学习总结附TF2.0代码实现)

# Transformer

谷歌发表的论文《Attention is all you need》中提出的一种新的深度学习架构transformer，其用于自然语言处理的重要性相信大家都有很深刻的认识，这里不再赘述，下文直接上干货。

# 1.Transformer详解

个人对transformer学习后，认为实现transformer总体流程如下：
 1. 输入部分：word embedding + positional ecoding；
 2. Multi-Headed Attention；
 3. Add and Layer normalization；
 4. Feed Forward；
 5. Decoder层；
 6. 输出层
 后文按照以上顺序，分别进行详细描述，每一部分理论完结都跟随相应代码实现。

## 1.1 transformer总体架构

和Seq2Seq模型一样，Transformer模型中也采用了 encoer-decoder 架构。但其结构相比于Seq2Seq更加复杂，论文中encoder层由6个encoder堆叠在一起，decoder层也一样。
每一个encoder和decoder的内部简版结构如下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191128215032693.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
对于encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。decoder也包含encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191128215050872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
现在我们知道了模型的主要组件，接下来我们看下模型的内部细节。首先，模型需要对输入的数据进行一个embedding操作，（也可以理解为类似w2c的操作），enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191128215058613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)

## 1.2 输入部分
transformer的输入部分使用的是word embedding和Positional Encoding的结合。为了解释输入序列中单词顺序，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和word embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019113014491261.png)
其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码。最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130145045768.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
为了让模型捕捉到单词的顺序信息，我们添加位置编码向量信息（POSITIONAL ENCODING），位置编码向量不需要训练，它有一个规则的产生方式（上图公式）。

如果我们的嵌入维度为4，那么实际上的位置编码就如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130145125145.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
那么生成位置向量需要遵循怎样的规则呢？

观察下面的图形，每一行都代表着对一个矢量的位置编码。因此第一行就是我们输入序列中第一个字的嵌入向量，每行都包含512个值，每个值介于1和-1之间。我们用颜色来表示1，-1之间的值，这样方便可视化的方式表现出来：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130145150364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
这是一个20个字（行）的（512）列位置编码示例。你会发现它咋中心位置被分为了2半，这是因为左半部分的值是一由一个正弦函数生成的，而右半部分是由另一个函数（余弦）生成。然后将它们连接起来形成每个位置编码矢量。

Positianal Encoding代码实现实例如下：
```javascript
def positional_encoding(pos, d_model):
    '''
    :param pos: 词在句子中的位置，句子上的维族；（i是d_model上的维度）
    :param d_model: 隐状态的维度，相当于num_units
    :return: 位置编码 shape=[1, position_num, d_model], 其中第一个维度是为了匹配batch_size
    '''
    def get_angles(position, i):
        # 这里的i相当于公式里面的2i或2i+1
        # 返回shape=[position_num, d_model]
        return position / np.power(10000., 2. * (i // 2.) / np.float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    # 2i位置使用sin编码，2i+1位置使用cos编码
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], tf.float32)
    return pos_encoding

# 演示positional_encoding
pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
```

## 1.3 Multi-Head Attention
接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正常理解的单词的一种思路，我们看个例子：
The animal didn't cross the street because it was too tired
这里的it到底代表的是animal还是street呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把it和animal联系起来

接下来我们看下详细的处理过程。

1、首先，self-attention会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是64低于embedding维度的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194620852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
那么Query、Key、Value这三个向量又是什么呢？这三个向量对于attention来说很重要，当你理解了下文后，你将会明白这三个向量扮演者什么的角色。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194641605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
2、计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点乘，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194651500.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
3、接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194659530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
4、下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194730440.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵Q与K相乘，乘以一个常数，做softmax操作，最后乘上V矩阵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194741154.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194752458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention。其实scaled dot-Product attention就是我们常用的使用点积进行相似度计算的attention，只是多除了一个（为K的维度）起到调节作用，使得内积不至于太大。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130194756645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
这篇论文更厉害的地方是给self-attention加入了另外一个机制，被称为“multi-headed” attention，该机制理解起来很简单，就是说不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195347312.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195355990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
这给我们留下了一个小的挑战，前馈神经网络没法输入8个矩阵呀，这该怎么办呢？所以我们需要一种方式，把8个矩阵降为1个，首先，我们把8个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘，最后得到一个最终的矩阵。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195410270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
这就是multi-headed attention的全部流程了，这里其实已经有很多矩阵了，我们把所有的矩阵放到一张图内看一下总体的流程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195418401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
多头attention（Multi-head attention）整个过程可以简述为：Query，Key，Value首先进过一个线性变换，然后输入到放缩点积attention（注意这里要做h次，其实也就是所谓的多头，每一次算一个头，而且每次Q，K，V进行线性变换的参数W是不一样的），然后将h次的放缩点积attention结果进行拼接，再进行一次线性变换得到的值作为多头attention的结果。可以看到，google提出来的多头attention的不同之处在于进行了h次计算而不仅仅算一次，论文中说到这样的好处是可以允许模型在不同的表示子空间里学习到相关的信息，后面还会根据attention可视化来验证。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195423663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
那么在整个模型中，是如何使用attention的呢？如下图，首先在编码器到解码器的地方使用了多头attention进行连接，K，V，Q分别是编码器的层输出（这里K=V）和解码器中都头attention的输入。其实就和主流的机器翻译模型中的attention一样，利用解码器和编码器attention来进行翻译对齐。然后在编码器和解码器中都使用了多头自注意力self-attention来学习文本的表示。Self-attention即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195434718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
对于使用自注意力机制的原因，论文中提到主要从三个方面考虑（每一层的复杂度，是否可以并行，长距离依赖学习），并给出了和RNN，CNN计算复杂度的比较。可以看到，如果输入序列n小于表示维度d的话，每一层的时间复杂度self-attention是比较有优势的。当n比较大时，作者也给出了一种解决方案self-attention（restricted）即每个词不是和所有词计算attention，而是只与限制的r个词去计算attention。在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195446426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
现在我们已经接触了attention的header，让我们重新审视我们之前的例子，看看例句中的“it”这个单词在不同的attention header情况下会有怎样不同的关注点（这里不同颜色代表attention不同头的结果，颜色越深attention值越大）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195452641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
当我们对“it”这个词进行编码时，一个注意力的焦点主要集中在“animal”上，而另一个注意力集中在“tired”（两个heads）
但是，如果我们将所有注意力添加到图片中，可能有点难理解：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130195456828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
Multi-Head Attention代码实现实例如下：
```javascript
'''*************** 第一部分: Scaled dot-product attention ***************'''
def scaled_dot_product_attention(q, k, v, mask):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    # query 和 Key相乘
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention =matmul_qk / tf.math.sqrt(dk)
    # 掩码mask
    if mask is not None:
        # 这里将mask的token乘以-1e-9，这样与attention相加后，mask的位置经过softmax后就为0
        # padding位置 mask=1
        scaled_attention += mask * -1e-9
    # 通过softmax获取attention权重, mask部分softmax后为0
    attention_weights = tf.nn.softmax(scaled_attention)  # shape=[batch_size, seq_len_q, seq_len_k]
    # 乘以value
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights

'''*************** 第二部分: Multi-Head Attention ***************'''
'''
multi-head attention包含3部分： - 线性层与分头 - 缩放点积注意力 - 头连接 - 末尾线性层
每个多头注意块有三个输入; Q（查询），K（密钥），V（值）。 它们通过第一层线性层并分成多个头。
注意:点积注意力时需要使用mask， 多头输出需要使用tf.transpose调整各维度。
Q，K和V不是一个单独的注意头，而是分成多个头，因为它允许模型共同参与来自不同表征空间的不同信息。
在拆分之后，每个头部具有降低的维度，总计算成本与具有全维度的单个头部注意力相同。
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # d_model必须可以正确分成多个头
        assert d_model % num_heads == 0
        # 分头之后维度
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头，将头个数的维度，放到seq_len前面 x输入shape=[batch_size, seq_len, d_model]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，根据q,k,v的输入，计算Q, K, V语义
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        # 分头
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # 通过缩放点积注意力层
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        # 把多头合并
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]
        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights

# 测试multi-head attention
temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))
output, att = temp_mha(y, y, y, None)
print(output.shape, att.shape)
```

## 1.4 Add and Layer normalization
在transformer中，每一个子层（self-attetion，ffnn）之后都会接一个残差模块，并且有一个Layer normalization
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130211821336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
在进一步探索其内部计算方式，我们可以将上面图层可视化为下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130211828544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
残差模块相信大家都很清楚了，这里不再讲解，主要讲解下Layer normalization。Normalization有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为0方差为1的数据。我们在把数据送入激活函数之前进行normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。
说到 normalization，那就肯定得提到 Batch Normalization。BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。
BN的具体做法就是对每一小批数据，在批这个方向上做归一化。如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130212251185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
可以看到，右半边求均值是沿着数据 batch_size的方向进行的，其计算公式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130212303189.png)
那么什么是 Layer normalization 呢？它也是归一化数据的一种方式，不过 LN 是在每一个样本上计算均值和方差，而不是BN那种在批方向计算均值和方差！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130212340647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
下面看一下 LN 的公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130212016617.png)
Layer normalization 代码实现实例如下：
```javascript
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x): # x shape=[batch_size, seq_len, d_model]
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
```


## 1.5 Feed Forward
模型经过multi-head attention后，经过一层feed forward层。模型中的前向反馈层，采用的是一种posion-wise feed-forward的方法，具体公式如下：FFN(x) = max(0, xW1 + b1)W2 + b2
此处方法容易理解，先对输入加一个全连接网络，之后使用Relu激活，之后再加一个全连接网络。

Feed Forward 代码实现实例如下：
```javascript
def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu),
        tf.keras.layers.Dense(d_model)
    ])
```

## 1.6 Encoder层
到这里为止就是全部encoders的内容了，如果把两个encoders叠加在一起就是这样的结构，在self-attention需要强调的最后一点是其采用了残差网络中的short-cut结构，目的是解决深度学习中的梯度弥散问题。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130221716218.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
整个 Encoder 过程代码实现实例如下：
```javascript
'''encoder layer:
每个编码层包含以下子层 - Multi-head attention（带掩码） - Point wise feed forward networks
每个子层中都有残差连接，并最后通过一个正则化层。残差连接有助于避免深度网络中的梯度消失问题。 
每个子层输出是LayerNorm(x + Sublayer(x))，规范化是在d_model维的向量上。Transformer一共有n个编码层。
'''
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training, mask):
        # multi head attention (encoder时Q = K = V)
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output)  # shape=[batch_size, seq_len, d_model]
        # feed forward network
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # shape=[batch_size, seq_len, d_model]
        return output2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model)  # shape=[batch_size, seq_len, d_model]
        self.pos_encoding = positional_encoding(max_seq_len, d_model)  # shape=[1, max_seq_len, d_model]
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training, mask):
        # 输入部分；inputs shape=[batch_size, seq_len]
        seq_len = inputs.shape[1]  # 句子真实长度
        word_embedding = self.emb(inputs)  # shape=[batch_size, seq_len, d_model]
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb= word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x  # shape=[batch_size, seq_len, d_model]

# 编码器测试
sample_encoder = Encoder(512, 2, 8, 1024, 5000, 200)
sample_encoder_output = sample_encoder(tf.random.uniform((64, 120)), False, None)
print(sample_encoder_output.shape)
```

## 1.7 Decoder层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191130225948981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMDc5MDIz,size_16,color_FFFFFF,t_70)
上图是transformer的一个详细结构，相比本文一开始结束的结构图会更详细些，接下来，我们会按照这个结构图讲解下decoder部分。
可以看到decoder部分其实和encoder部分大同小异，不过在最下面额外多了一个masked mutil-head attetion，这里的mask也是transformer一个很关键的技术，我们一起来看一下。
### Mask
mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。
其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。
### Padding Mask
什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。
具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！
而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。
### Sequence mask
文章前面也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。
那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。
对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个mask相加作为attn_mask。
其他情况，attn_mask 一律等于 padding mask。
编码器通过处理输入序列启动。然后将顶部编码器的输出转换为一组注意向量k和v。每个解码器将在其“encoder-decoder attention”层中使用这些注意向量，这有助于解码器将注意力集中在输入序列中的适当位置：
![在这里插入图片描述](https://img-blog.csdnimg.cn/201911302301063.gif)
完成编码阶段后，我们开始解码阶段。解码阶段的每个步骤从输出序列（本例中为英语翻译句）输出一个元素。
以下步骤重复此过程，一直到达到表示解码器已完成输出的符号。每一步的输出在下一个时间步被送入底部解码器，解码器像就像我们对编码器输入所做操作那样，我们将位置编码嵌入并添加到这些解码器输入中，以表示每个字的位置。

整个Decoder 过程代码实现实例如下：
```javascript
# padding mask
def create_padding_mask(seq):
    '''为了避免输入中padding的token对句子语义的影响，需要将padding位mark掉，
    原来为0的padding项的mask输出为1; encoder和decoder过程都会用到'''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 扩充维度以便于使用attention矩阵;seq输入shape=[batch_size, seq_len]；输出shape=[batch_siz, 1, 1, seq_len]
    return seq[:, np.newaxis, np.newaxis, :]

# look-ahead mask
def create_look_ahead_mask(size):
    '''用于对未预测的token进行掩码 这意味着要预测第三个单词，只会使用第一个和第二个单词。
    要预测第四个单词，仅使用第一个，第二个和第三个单词，依此类推。只有decoder过程用到'''
    # 产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape=[seq_len, seq_len]
 
def create_mask(inputs, targets):
    # 编码器只有padding_mask
    encoder_padding_mask = create_padding_mask(inputs)
    # 解码器decoder_padding_mask,用于第二层multi-head attention
    decoder_padding_mask = create_padding_mask(inputs)
    # seq_mask mask掉未预测的词
    seq_mask = create_look_ahead_mask(tf.shape(targets)[1])
    # decoder_targets_padding_mask 解码层的输入padding mask
    decoder_targets_padding_mask = create_padding_mask(targets)
    # 合并解码层mask，用于第一层masked multi-head attention
    look_ahead_mask = tf.maximum(decoder_targets_padding_mask, seq_mask)
    return encoder_padding_mask, look_ahead_mask, decoder_padding_mask

'''
decoder layer:
每个编码层包含以下子层： - Masked muti-head attention（带padding掩码和look-ahead掩码
- Muti-head attention（带padding掩码）value和key来自encoder输出，
query来自Masked muti-head attention层输出 - Point wise feed forward network
每个子层中都有残差连接，并最后通过一个正则化层。残差连接有助于避免深度网络中的梯度消失问题。
每个子层输出是LayerNorm(x + Sublayer(x))，规范化是在d_model维的向量上。Transformer一共有n个解码层。
当Q从解码器的第一个注意块接收输出，并且K接收编码器输出时，注意权重表示基于编码器输出给予解码器输入的重要性。
换句话说，解码器通过查看编码器输出并自我关注其自己的输出来预测下一个字。
ps：因为padding在后面所以look-ahead掩码同时掩padding
'''
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        # masked multi-head attention: Q = K = V
        att_out1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att_out1 = self.dropout1(att_out1, training=training)
        att_out1 = self.layernorm1(inputs + att_out1)
        # multi-head attention: Q=att_out1, K = V = encoder_out
        att_out2, att_weight2 = self.mha2(att_out1, encoder_out, encoder_out, padding_mask)
        att_out2 = self.dropout2(att_out2, training=training)
        att_out2 = self.layernorm2(att_out1 + att_out2)
        # feed forward network
        ffn_out = self.ffn(att_out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        output = self.layernorm3(att_out2 + ffn_out)
        return output, att_weight1, att_weight2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.seq_len = tf.shape
        self.d_model = d_model
        self.num_layers = num_layers
        self.word_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = inputs.shape[1]
        attention_weights = {}
        word_embedding = self.word_embedding(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x, att1, att2 = self.decoder_layers[i](x, encoder_out, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att_w2'.format(i + 1)] = att2
        return x, attention_weights

# 解码器测试
sample_decoder = Decoder(512, 2, 8, 1024, 5000, 200)
sample_decoder_output, attn = sample_decoder(tf.random.uniform((64, 100)),
                                             sample_encoder_output, False, None, None)
print(sample_decoder_output.shape)
print(attn['decoder_layer1_att_w2'].shape)
```

## 1.8 Transformer
当decoder层全部执行完毕后，怎么把得到的向量映射为我们需要的词呢，很简单，只需要在结尾再添加一个全连接层和softmax层，假如我们的词典是1w个词，那最终softmax会输入1w个词的概率，概率值最大的对应的词就是我们最终的结果。

最后整个transformer 过程代码实现实例如下：
```javascript
'''Transformer包含编码器、解码器和最后的线性层，解码层的输出经过线性层后得到Transformer的输出'''
class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inputs, targets, training, encoder_padding_mask,
             look_ahead_mask, decoder_padding_mask):
        # 首先encoder过程，输出shape=[batch_size, seq_len_input, d_model]
        encoder_output = self.encoder(inputs, training, encoder_padding_mask)
        # 再进行decoder, 输出shape=[batch_size, seq_len_target, d_model]
        decoder_output, att_weights = self.decoder(targets, encoder_output, training,
                                                   look_ahead_mask, decoder_padding_mask)
        # 最后映射到输出层
        final_out = self.final_layer(decoder_output) # shape=[batch_size, seq_len_target, target_vocab_size]
        return final_out, att_weights

# transformer测试
sample_transformer = Transformer(
num_layers=2, d_model=512, num_heads=8, dff=1024,
input_vocab_size=8500, target_vocab_size=8000, max_seq_len=120
)
temp_input = tf.random.uniform((64, 62))
temp_target = tf.random.uniform((64, 26))
fn_out, att = sample_transformer(temp_input, temp_target, training=False,
                              encoder_padding_mask=None,
                               look_ahead_mask=None,
                               decoder_padding_mask=None,
                              )
print(fn_out.shape)
print(att['decoder_layer1_att_w1'].shape)
print(att['decoder_layer1_att_w2'].shape)
```

至此，整个Transformer的构建过程结束！
