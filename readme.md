## Ancient Chinese Poems Generation

&emsp;&emsp;NLP task: Generation of *ancient Chinese poems* (ACPG) 

> Notice: The follwing contents will be a mixture of Chinese & English cuz I'm too lazy to translate the original Chinese docs into English:man:

#### Instructions about ancient Chinese poems

&emsp;&emsp;This project aims to generate a complete poem according to a given first line. The poems we focused on are the ones of four lines with seven characters each line, which are also called "Qijue(七言绝句)". Other requirements of Qijue include the rhym of the poem. The finals(which means the simple or compound vowel of a Chinese syllable) of the last character of the second and the fourth line should be the same one. In addition, the third line can also rhym, which would be better.

#### Requirements

* python3 + pytorch0.4/1.0 + cuda8/9
* pypinyin module, for choosing the rhyming character

#### Docs

&emsp;&emsp;Implementation: We use the common generating model Seq2Seq for this task. The pairs we used as the traning data were processed in the formation of｛第一句，第二句｝，｛第一句+第二句，第三句｝，｛第一句+第二句+第三句，第四句｝. Our encoder and decoder use undirectional lstms with pretrained word-embeddings([pretrained with The Si Ku Quan Shu(四库全书)](https://github.com/Embedding/Chinese-Word-Vectors#various-domains)). We introduced the attention mechanism(concat attention) into our decoder, and the result became the best(meanwhile the loss was the lowest) when it came to around 15 rounds.

&emsp;&emsp;Generating Stage: Each time we generate a poem, we use the first line as input and generates a second line. Then use the first two lines as input and generates a third line. Finally we use the first three lines as input and generates a final line. Meanwhile, considering the matter of rhyming, we will generate two versions(nonrhyme & rhyme) each time we execute the test phase.

&emsp;&emsp;Version of Rhyming: We needs the module pypinyin, which can give us the final of a character directly(see the following codes). At the stage of generating, choose the model of rhyming, then the model would find out the several indexes of the most possible output words on the vocabulary when decoding the last character of each line. Then we would find if there's a word that rhymes with the rhyme. If we find one ryhmes, we use it as the prediction. Otherwise we would just use the one that's the most possible. Note: 对于韵脚的选择：生成第2、3句时，韵脚均选取上一句的句尾字；生成第四句选取第二句的句尾字。备选项的个数CANDIDATE也是根据各句对于押韵的不同要求程度定义，第二句为5，第三句为3，第四句为16.
```python
>>> lazy_pinyin('难', style=Style.FINALS, strict=False)[0]
>>> 'an'
```
#### Other Matters

&emsp;&emsp;Worked with my teammate Ziyi Yang. I suspect that the results might be better if we introduce the full stops & commas to the corpus, but we didn't give it a try. Morever, poetry-generation2 is the implementation with better results while the model in master directory doesn't work well, about which I still have no idea. In fact, the master one uses Bi-directonal lstms while the second one with undirectonal lstms, and the master one is trained in batches while the second one is trained one by one. I can't work out where the matter is.

#### Results

&emsp;&emsp;Some good results:

Rhythm：

>&emsp;&emsp;长安少女踏春阳，百草千花满路长。谁道东家红不去，淡云春水看春阳。  
>&emsp;&emsp;一声天边断雁哀，清寒千古更无台。秋风吹作千山事，不用西风一水来。  
>&emsp;&emsp;湖上秋山翠作堆，千云不日不徘徊。故教亭上人间事，一夜扁舟过眼回。  
>&emsp;&emsp;零雨崇朝不下楼，楚人只在水中洲。沙头欲问无人事，一夜江舟入夜愁。  
>&emsp;&emsp;斑斑血洒哭诗章，父老苍生作事长。莫说千年无一事，眼中犹恨是无阳。  
>&emsp;&emsp;宝剑翩翩赋远游，萧舟风月夜秋愁。金门万里无人去，遥指江头夜夜楼。  

Nonrhym：

>&emsp;&emsp;见羊疑是已叱石，百人飞来如不知，走雨再生回首去，夜深犹在望云来。  
>&emsp;&emsp;乞得衰身出瘴烟，春来一饭说余神，君知不是无人事，忍把青间也不知。  
>&emsp;&emsp;六月浑如九月清，相花如月水如明，夜来雨后无人梦，错认芭蕉叶上声。  
#### Reference

* [Qixin Wang, Tianyi Luo, Dong Wang, and Chao Xing. 2016. Chinese song iambics generation with neural attention-based model.CoRR, abs/1604.06274.](https://www.researchgate.net/publication/301878077_Chinese_Song_Iambics_Generation_with_Neural_Attention-based_Model)

* [Github: Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
