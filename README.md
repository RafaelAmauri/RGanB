# R-Gan-B, a deep learning image colorizer!

By: Me and [Sandor Scoggin](https://github.com/sandorscog)

## Summary 
This project is a simple and naive implementation of a [Generative Adversarial Network](https://www.youtube.com/watch?v=Sw9r8CL98N0) that takes a black-and-white image and outputs an RGB version of that image.

This is my very first deep learning project, and I know that this isn’t the optimal approach. However, I wanted to see what I could achieve with a very rudimentary understanding of deep learning and GAN models.


## Overview of the method

This project uses a GAN setup consisting of a generator and a discriminator.

The first step is to convert the black-and-white image into the [LAB colorspace](https://en.wikipedia.org/wiki/CIELAB_color_space), which divides the image into three channels:

    L (Lightness): Represents the brightness of each pixel and is very similar to the original black-and-white image.
    A: Contains chromatic information.
    B: Contains the other set of chromatic information.


We do this because the L channel is pretty much the same as the black-and-white image, meaning we already have one of the three channels. This approach also makes it easier to train a network that only has to predict two components (A and B) rather than three (R, G, and B).

After converting the black-and-white images to into LAB images, we train the generator model to output the A and B channels for a given L channel.


## Generator

The generator is a [U-Net](https://en.wikipedia.org/wiki/U-Net) with skip connections that help preserve spatial details lost during downsampling. The architecture of the generator is laid out below::

![Generator Architecture](assets/generator.png)

We chose to use an U-Net mainly because it was a way for me to learn how to implement skip connections in a Convolutional Neural Network. Also, U-Nets are known for performing well in image colorization tasks. :)

## Discriminator

The discriminator model is a [Patch-GAN](https://jimchopper.medium.com/what-is-patchgan-e7e17a1c479a) model. The main reason for choosing Patch-GAN over a simple Convolutional Neural Network is that Patch-GAN divides the image into patches and provides a unique output for each of them rather than a single output for the entire image - This gives the generator localized feedback on **which** parts of the generated image appear fake, allowing it to focus on specific regions instead of adjusting the entire image at once.

The architecture for our Patch-Gan was graciously contributed by [junyanz](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We thank the original author for making his code open-source and available to others :)


## Loss Functions

For the discriminator, we use BCEWithLogits instead of a regular Binary Cross Entropy loss function because Binary Cross Entropy applies a sigmoid function at the output to convert the logits into probabilities before computing the Cross-Entropy. However, this can push the outputs too close to 0 or 1 if the logits are too large, which could possibly lead to over/underflow issues when performing the Cross-Entropy calculation. 

On the other hand, BCEWithLogits combines the sigmoid and the Cross-Entropy into a single function rather than treating them as separate steps, which avoids the over/underflow problem.

The loss function for the Discriminator is defined as: 

$LossD = [ BCEWithLogitsLoss(Discriminator(Images_{fake}), 0) + BCEWithLogits(Discriminator(Images_{real}), 1) ] / 2$

The loss function for the Generator is the adversarial loss and an L1 regularization term. The L1 regulation term is scaled by a factor of 100, because the L1 loss computed over a large number of pixels tends to yield small values.

The Generator's loss function is defined as:

$LossG = BCEWithLogitsLoss(Discriminator(Images_{fake}), 1) + 100 * L1Loss(AB_{fake}, AB_{real})$

## Results and Areas of Improvement

Since this is my very first deep learning project, I obtained a mix of good and bad results. They are shown below:

| **Category**         | **Input**                                          | **AI-Generated**                                   | **Ground-Truth**                                    |
|----------------------|----------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| **Good**     | ![onibus_gray](assets/onibus_grayscale.jpeg)       | ![onibus_g](assets/onibus_colorized.jpeg)           | ![onibus_gt](assets/onibus_groundtruth.jpeg)         |
| **Good**  | ![geladeira_gray](assets/geladeira_entrada.jpeg)     | ![geladeira_g](assets/geladeira_g.jpeg)             | ![geladeira_gt](assets/geladeira_gt.jpeg)            |
| **Bad**      | ![frutas_gray](assets/fruta_grayscale.jpeg)          | ![frutas_g](assets/fruta_colorized.jpeg)            | ![frutas_gt](assets/fruta_groundtruth.jpeg)          |
| **Bad**      | ![menino_gray](assets/menino_entrada.jpeg)           | ![menino_g](assets/menino_g.jpeg)                   | ![menino_gt](assets/menino_gt.jpeg)                  |


There are many aspects of the code that could be improved. The main one is the loss function for the generator, which is currently scaled by 100. We could see an improvement if we scale it down to something like 30 or 20. As it stands, the regularization term is completely dominating the Generator's loss, and it is focusing on improving that rather than the adversarial loss.