# Assignment - 1B

## What are Channels and Kernels (according to EVA)?

### Channels

Channels are components of an image that represent a particular feature or attribute present in a data set. Channels can represent on any feature of the image. Most commonly used channels in images are of color separations like: RGB, CMYK, HSV, YCbCr etc. In certain MRI kind of scan results even 7-8 color separation channels can be used. However, in the deep learning context the channels are generalized to any feature. For example, when you detect images in an image this introduces a channel consisting of edges in the images. A filter or kernel is applied on one or more of the channels of the images to bring out features in an image as channels. 

### Kernels

Kernels are filters that are applied on a collection of channels to accentuate a specific attribute present in the image. The operation is mostly of convolution type. Essentially, kernels contain the attributes that helps in extracting the specific features in an image. Channels are provided as inputs and a new sets of channels are produced as outputs to the convolution operations. 

## Why should we only (well mostly) use 3x3 Kernels?

**Variable Target Receptive Field**: For an object to be detected and in an image, the object must be fully visible in the receptive field of the image. This essentially means the kernel which is in use must be aware of the size of the object apriori. Or the kernel must be the largest of the objects to be included or excluded in a feature detection phase. This is not practically, feasible in most real life cases. 

**Reduction in Computational Complexity**: Number of parameters used in a kernel is square of the number of pixels covered. A $3 \times3 $ has 9 parameters while $ 5 \times 5$ has 25. Using $ 3 \times 3 $ twice provides the same receptive field of a $ 3 \times 3$ kernel thus reducing some computational complexities. Although, the receptive field is same there may be some loss of generality in using the lesser parameters. However, most cases such losses are not significant. But for images where the complexities are high with rapidly changing features. the usage of higher dimensional kernel may help. But, this can lead to additional computing cycles. 

## How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

Each operation leads to a reduction of 2 values from the input. Hence, it will take 99 steps. The details of the calculation is shown below. 

|   Input   | kernel |  Output   |
| :-------: | :----: | :-------: |
| 199 x 199 |  3x3   | 197 x 197 |
| 197 x 197 |  3x3   | 195 x 195 |
| 195 x 195 |  3x3   | 193 x 193 |
| 193 x 193 |  3x3   | 191 x 191 |
| 191 x 191 |  3x3   | 189 x 189 |
| 189 x 189 |  3x3   | 187 x 187 |
| 187 x 187 |  3x3   | 185 x 185 |
| 185 x 185 |  3x3   | 183 x 183 |
| 183 x 183 |  3x3   | 181 x 181 |
| 181 x 181 |  3x3   | 179 x 179 |
| 179 x 179 |  3x3   | 177 x 177 |
| 177 x 177 |  3x3   | 175 x 175 |
| 175 x 175 |  3x3   | 173 x 173 |
| 173 x 173 |  3x3   | 171 x 171 |
| 171 x 171 |  3x3   | 169 x 169 |
| 169 x 169 |  3x3   | 167 x 167 |
| 167 x 167 |  3x3   | 165 x 165 |
| 165 x 165 |  3x3   | 163 x 163 |
| 163 x 163 |  3x3   | 161 x 161 |
| 161 x 161 |  3x3   | 159 x 159 |
| 159 x 159 |  3x3   | 157 x 157 |
| 157 x 157 |  3x3   | 155 x 155 |
| 155 x 155 |  3x3   | 153 x 153 |
| 153 x 153 |  3x3   | 151 x 151 |
| 151 x 151 |  3x3   | 149 x 149 |
| 149 x 149 |  3x3   | 147 x 147 |
| 147 x 147 |  3x3   | 145 x 145 |
| 145 x 145 |  3x3   | 143 x 143 |
| 143 x 143 |  3x3   | 141 x 141 |
| 141 x 141 |  3x3   | 139 x 139 |
| 139 x 139 |  3x3   | 137 x 137 |
| 137 x 137 |  3x3   | 135 x 135 |
| 135 x 135 |  3x3   | 133 x 133 |
| 133 x 133 |  3x3   | 131 x 131 |
| 131 x 131 |  3x3   | 129 x 129 |
| 129 x 129 |  3x3   | 127 x 127 |
| 127 x 127 |  3x3   | 125 x 125 |
| 125 x 125 |  3x3   | 123 x 123 |
| 123 x 123 |  3x3   | 121 x 121 |
| 121 x 121 |  3x3   | 119 x 119 |
| 119 x 119 |  3x3   | 117 x 117 |
| 117 x 117 |  3x3   | 115 x 115 |
| 115 x 115 |  3x3   | 113 x 113 |
| 113 x 113 |  3x3   | 111 x 111 |
| 111 x 111 |  3x3   | 109 x 109 |
| 109 x 109 |  3x3   | 107 x 107 |
| 107 x 107 |  3x3   | 105 x 105 |
| 105 x 105 |  3x3   | 103 x 103 |
| 103 x 103 |  3x3   | 101 x 101 |
| 101 x 101 |  3x3   |  99 x 99  |
|  99 x 99  |  3x3   |  97 x 97  |
|  97 x 97  |  3x3   |  95 x 95  |
|  95 x 95  |  3x3   |  93 x 93  |
|  93 x 93  |  3x3   |  91 x 91  |
|  91 x 91  |  3x3   |  89 x 89  |
|  89 x 89  |  3x3   |  87 x 87  |
|  87 x 87  |  3x3   |  85 x 85  |
|  85 x 85  |  3x3   |  83 x 83  |
|  83 x 83  |  3x3   |  81 x 81  |
|  81 x 81  |  3x3   |  79 x 79  |
|  79 x 79  |  3x3   |  77 x 77  |
|  77 x 77  |  3x3   |  75 x 75  |
|  75 x 75  |  3x3   |  73 x 73  |
|  73 x 73  |  3x3   |  71 x 71  |
|  71 x 71  |  3x3   |  69 x 69  |
|  69 x 69  |  3x3   |  67 x 67  |
|  67 x 67  |  3x3   |  65 x 65  |
|  65 x 65  |  3x3   |  63 x 63  |
|  63 x 63  |  3x3   |  61 x 61  |
|  61 x 61  |  3x3   |  59 x 59  |
|  59 x 59  |  3x3   |  57 x 57  |
|  57 x 57  |  3x3   |  55 x 55  |
|  55 x 55  |  3x3   |  53 x 53  |
|  53 x 53  |  3x3   |  51 x 51  |
|  51 x 51  |  3x3   |  49 x 49  |
|  49 x 49  |  3x3   |  47 x 47  |
|  47 x 47  |  3x3   |  45 x 45  |
|  45 x 45  |  3x3   |  43 x 43  |
|  43 x 43  |  3x3   |  41 x 41  |
|  41 x 41  |  3x3   |  39 x 39  |
|  39 x 39  |  3x3   |  37 x 37  |
|  37 x 37  |  3x3   |  35 x 35  |
|  35 x 35  |  3x3   |  33 x 33  |
|  33 x 33  |  3x3   |  31 x 31  |
|  31 x 31  |  3x3   |  29 x 29  |
|  29 x 29  |  3x3   |  27 x 27  |
|  27 x 27  |  3x3   |  25 x 25  |
|  25 x 25  |  3x3   |  23 x 23  |
|  23 x 23  |  3x3   |  21 x 21  |
|  21 x 21  |  3x3   |  19 x 19  |
|  19 x 19  |  3x3   |  17 x 17  |
|  17 x 17  |  3x3   |  15 x 15  |
|  15 x 15  |  3x3   |  13 x 13  |
|  13 x 13  |  3x3   |  11 x 11  |
|  11 x 11  |  3x3   |   9 x 9   |
|   9 x 9   |  3x3   |   7 x 7   |
|   7 x 7   |  3x3   |   5 x 5   |
|   5 x 5   |  3x3   |   3 x 3   |
|   3 x 3   |  3x3   |   1 x 1   |
| No of steps:|      | 99        |
----------------------------------
