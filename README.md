# deeppose-1

1. 서버 들어가기
```
>> ssh (아이디)@alpha.inu.ac.k
```

2. 가상환경 만들기
```
>> virtualenv -p python3 jy
```

3. 가상환경들어가기
```
>> source jy/bin/activate
```

4. git clone 하기
```mkdir src```

```
cd src
```

```
>> git clone https://github.com/ys7yoo/deeppose.git
```

7. 우리 서버 weight 받기
```
cd deeppose
```
  
```
mkdir weights
```

```
cd weights
```
```
>> wget http://smart.inu.ac.kr/weights/bvlc_alexnet.tf
```

8. dataset 받기
```
cd datasets
```
```
./download_lsp.sh
```


# 가상환경에서 필요한거 다운

1. 
```
>> ln -s /usr/local/lib/python3.5/dist-packages/cv2.so ~/(가상환경이름)/lib/python3.5/site-packages/cv2.so
```
> 이 방법으로 다운안되면 `pip3 install opencv-python`로 다운


2. tensorflow설치 tensorflow version 1.4.1로 설치 
```
>> pip3 install tensorflow-gpu==1.4.1
```

3. 나머지 깔기
```
>> pip install --upgrade chainer numpy tqdm scipy
```

4. 원진 0번,  주영 1번, 성우 2번, 소연 3번
```
>> CUDA_VISIBLE_DEVICES=1 bash examples/train_lsp_alexnet_imagenet_small.sh
```

5. deeppose 디렉토리에서
```
>> export PYTHONPATH=`pwd`
```

6. small 데이터 트레이닝
```
>> bash examples/train_lsp_alexnet_imagenet_small.sh 
```

7. 서버 내에서 training된 weight 가져오기 
```
>> tar xvfz /var/lsp_alexnet_imagenet.tar.gz 
```

8. 원본 이미지넷 테스트
```
python tests/test_snapshot.py lsp out/lsp_alexnet_imagenet/checkpoint-50000
```



# restore하기

>>> 아직 시행착오중

1번>>
test_restore

import tensorflow as tf

saver=tf.train.import_meta_graph("out/lsp_alexnet_imagenet_small/checkpoint-150000.meta")

with tf.Session() as sess:
saver.restore(sess, "out/lsp_alexnet_imagenet_small/checkpoint-150000.data-00000-of-00001")

2번>>

#/model/test.py

with tf.Session(graph=g) as sess :

    # Saver instance 를 생성한다.
    # Saver.restore(sess, ckpt_path)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    # Saver.restore()
    # args : tf.Session, job`s checkpoint file path
    # return : None

    ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("saved"))

