# Assignment 7A

As can be seen the last average pool operation has a stride of 7 and not 1, very similar to maxpool on a 2x2 kernel has a default stride of 2. However, the jump is 224. This jump value is what has been mentioned as the receptive field in the paper. The actual receptive field is much larger. 
|n|k|p|s|r|j|
|-:|-:|-:|-:|-:|-:|
|228|7|0|2|1|1|
|112|3|0|2|7|2|
|56|3|1|1|11|4|
|56|3|0|2|19|4|
|28|3|0|2|27|8|
|14|3|0|2|43|16|
|7|7|0|7|75|32|
|1| | | |267|224|
