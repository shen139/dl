# Sequence Prediction with Keras #

```
$ python dense.py
```

## x * sin(x) ##
![xsinx](https://raw.githubusercontent.com/shen139/dl/master/keras/sequence_prediction/img/xsinx.jpg)

```
------------ x * sin(x) ------------
Training ... (Press CTRL+C to stop)
Epoch 1/5
170/170 [==============================] - 0s - loss: 0.9504
Epoch 2/5
170/170 [==============================] - 0s - loss: 0.4963
Epoch 3/5
170/170 [==============================] - 0s - loss: 0.2294
Epoch 4/5
170/170 [==============================] - 0s - loss: 0.0912
Epoch 5/5
170/170 [==============================] - 0s - loss: 0.0347
...
170/170 [==============================] - 0s - loss: 6.5138e-15      
Epoch 495/500      
170/170 [==============================] - 0s - loss: 6.2815e-15         
Epoch 496/500      
170/170 [==============================] - 0s - loss: 6.5024e-15         
Epoch 497/500      
170/170 [==============================] - 0s - loss: 6.2422e-15         
Epoch 498/500      
170/170 [==============================] - 0s - loss: 6.3031e-15         
Epoch 499/500      
170/170 [==============================] - 0s - loss: 6.6821e-15         
Epoch 500/500      
170/170 [==============================] - 0s - loss: 7.2068e-15         
Predicting...      
Last 5 values: [-17.44700408 -36.87622374 -47.58375927 -46.73387999 -34.29780523]          
Next -> [-13.11874078  11.73487105  34.18167285  48.61543339  51.3046351 
  41.3545167   20.98403554  -4.96432036 -30.17461159 -48.39467809        
 -54.98652396 -48.10514937 -29.20685548  -2.74863214  24.8613709         
  46.81589765  57.5866043   54.31524584  37.56753709  11.25233335        
 -18.2886289  -43.80851162 -58.93316851 -59.7526756  -45.82919141        
 -20.362212    10.54342809  39.34027951  58.88163819  64.19460656        
  53.74385631]
     
```


---


## cox(x) ##
![cosx](https://github.com/shen139/dl/blob/master/keras/sequence_prediction/img/cosx.jpg)

```
...
170/170 [==============================] - 0s - loss: 3.0620e-14         
Epoch 495/500      
170/170 [==============================] - 0s - loss: 2.7655e-14         
Epoch 496/500      
170/170 [==============================] - 0s - loss: 2.6893e-14         
Epoch 497/500      
170/170 [==============================] - 0s - loss: 2.0260e-14         
Epoch 498/500      
170/170 [==============================] - 0s - loss: 2.4997e-14         
Epoch 499/500      
170/170 [==============================] - 0s - loss: 2.6527e-14         
Epoch 500/500      
170/170 [==============================] - 0s - loss: 3.0053e-14         
Predicting...      
Last 5 values: [-0.93010041 -0.64014434 -0.1934586   0.30059254  0.72104815]     
Next -> [ 0.96496603  0.97262639  0.74215387  0.32997651 -0.16299108 -0.61605239 
 -0.91828278 -0.99568553 -0.82930981 -0.45989005  0.02212665  0.49872617 
  0.85322018  0.9988158   0.89986675  0.58059912  0.11917998 -0.37141823 
 -0.77108008 -0.98195503 -0.95241293 -0.6896867  -0.25810185  0.23667588 
  0.67350753  0.94544011  0.98589645  0.78497163  0.39185705 -0.09719726 
 -0.56245354]    
```


---


## parabola ##
![parabola](https://raw.githubusercontent.com/shen139/dl/master/keras/sequence_prediction/img/parabola.jpg)

```
...
170/170 [==============================] - 0s - loss: 1.0187e-14         
Epoch 495/500      
170/170 [==============================] - 0s - loss: 9.4296e-15         
Epoch 496/500      
170/170 [==============================] - 0s - loss: 9.4174e-15         
Epoch 497/500      
170/170 [==============================] - 0s - loss: 9.2342e-15         
Epoch 498/500      
170/170 [==============================] - 0s - loss: 9.5804e-15         
Epoch 499/500      
170/170 [==============================] - 0s - loss: 9.4390e-15         
Epoch 500/500      
170/170 [==============================] - 0s - loss: 9.5032e-15         
Predicting...      
Last 5 values: [ 2256.25  2304.    2352.25  2401.    2450.25]  
Next -> [ 2500.          2550.24984479  2600.99992156  2652.25023031  2703.99972796        
  2756.25005364  2809.00016427  2862.25005984  2916.00033641  2970.25039792      
  3025.00024438  3080.25062084  3136.00003719  3192.2505796   3249.00016189      
  3306.2505722   3364.00002241  3422.25074768  3481.00066185  3540.25095701      
  3600.00059009  3660.25045514  3721.00070119  3782.25147724  3844.00069714      
  3906.25119209  3969.00117397  4032.25153685  4096.00138664  4160.25102139      
  4225.00163317]     
```


---


## simple ##
![simple](https://raw.githubusercontent.com/shen139/dl/master/keras/sequence_prediction/img/simple.jpg)

```
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 495/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 496/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 497/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 498/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 499/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Epoch 500/500      
20/20 [==============================] - 0s - loss: 5.2652e-15 
Predicting...      
Last 5 values: [ 45.  46.  47.  48.  49.]    
Next -> [ 50.00000244  51.00001073  52.00000441  53.00000393  54.0000093 
  55.0000059   56.00000834  57.00000495  58.00001031  59.00000399        
  60.0000006   61.00000596  62.00000548  63.00001085  64.00000453        
  65.00000697  66.0000065   67.0000031   68.00001138  69.00001967        
  70.00001919  71.0000158   72.00002116  73.00002068  74.00002313        
  75.00002265  76.00002217  77.0000217   78.00001538  79.0000149         
  80.00002027]      
```


---


## 3x ##
![3x](https://raw.githubusercontent.com/shen139/dl/master/keras/sequence_prediction/img/3x.jpg)

```
20/20 [==============================] - 0s - loss: 4.9144e-15 
Epoch 495/500      
20/20 [==============================] - 0s - loss: 4.9144e-15 
Epoch 496/500      
20/20 [==============================] - 0s - loss: 4.9144e-15 
Epoch 497/500      
20/20 [==============================] - 0s - loss: 4.7531e-15 
Epoch 498/500      
20/20 [==============================] - 0s - loss: 4.7531e-15 
Epoch 499/500      
20/20 [==============================] - 0s - loss: 4.7531e-15 
Epoch 500/500      
20/20 [==============================] - 0s - loss: 4.7531e-15 
Predicting...      
Last 5 values: [ 135.  138.  141.  144.  147.]         
Next -> [ 149.99999857  153.0000059   155.99998695  158.99999428  161.99999285   
  165.00000018  167.99998999  170.99998856  173.99998713  176.99998569   
  179.99998426  182.99996531  185.99999017  188.99998873  191.99997854   
  194.99997711  197.99999321  201.00000054  203.99999034  206.99998891   
  209.99999624  213.0000211   216.00000215  219.00001824  221.99998176   
  225.0000329   227.9999789   230.99999499  234.00002861  237.00000966   
  240.0000608 ]     
```


