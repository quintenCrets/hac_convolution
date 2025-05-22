# CONVULUTION

This project contains code for the jetson nano, running cuda. It takes an .png image as input and outputs it to "outpu_file.png".
There are multiple calculation that can be done like edgedetection pooling and grayscaling.

## all options

- edge_detection
- gray_scale
- average_pooling
- max_pooling
- min_pooling

## examples

```
    ./main image.png edge_detection min_pooling
    ./main image.png gray_scale average_pooling edge_detection
```

