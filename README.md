# InSituNet

PyTorch implementation of the deep learning model introduced in our SciVis 2019 paper "InSituNet: Deep Image Synthesis for Parameter Space Exploration of Ensemble Simulations".

#### Comparison of InSituNet trained with different loss functions
![Results of different loss functions](https://github.com/hewenbin/insitu_net/blob/master/images/compare_loss.png)

#### Example visualization images generated by InSituNet
![Example visualization images generated by InSituNet](https://github.com/hewenbin/insitu_net/blob/master/images/prediction_images.png)

#### The way of training

```
unzip mpas_sub.zip 
mv mpas_sub/train/params_sub.npy mpas_sub/train/params.npy
mv mpas_sub/test/params_sub.npy mpas_sub/test/params.npy
mv mpas_sub datasets
cd model 
python main.py --root ../datasets --dsp 1 --gan-loss vanilla --gan-loss-weight 1e-2
```

#### The way of evaluation
```
python eval.py --root ../datasets \
               --dsp 1 \
               --resume {the path to the saved tar model} \
               --id {image id in the testing dataset}
```
Then the ground truth, predicted, and difference images are saved. 