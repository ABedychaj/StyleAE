# StyleAutoEncoder

Implementation of paper:

`"StyleAutoEncoder for manipulating image attributes using pre-trained StyleGAN"`

## Abstract

Deep conditional generative models are excellent tools for creating high-quality images and editing their attributes. However, training modern generative models from scratch is very expensive and requires large computational resources. In this paper, we introduce StyleAutoEncoder (StyleAE), a lightweight AutoEncoder module, which works as a plugin for pre-trained generative models and allows for manipulating the requested attributes of images. The proposed method offers a cost-effective solution for training deep generative models with limited computational resources, making it a promising technique for a wide range of applications. 
We evaluate StyleAE by combining it with StyleGAN, which is currently one of the top generative models. 
Our experiments demonstrate that StyleAE is at least as effective in manipulating image attributes as the state-of-the-art algorithms based on invertible normalizing flows. However, it is simpler, faster, and gives more freedom in designing neural architecture.