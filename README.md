# deeppose-1

1. ssh (아이디)@alpha.inu.ac.kr 서버 들어가기

2. virtualenv -p python3 (jy) 가상환경 만들기

3. 가상환경들어가서 git clone (deeppose)

4. vi ~/.bashrc 들어가서  아래추가

  export PATH=$PATH:/usr/local/cuda/bin
  
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
  
  export LD_INCLUDE_PATH=$LD_INCLUDE_PATH:/usr/local/cuda/include
  
  [vi>>> esc누른후 :q 그냥 나가기/:wq 저장 후 나가기]
  

5. mkdir weights

cd weights

wget http://smart.inu.ac.kr/weights/bvlc_alexnet.tf



6. cd datasets

./download_lsp.sh   # to get LSP dataset

./download_mpii.sh  # to get MPII dataset(아직)

cd ..



7. export PYTHONPATH=`pwd`

python datasets/lsp_dataset.py

python datasets/mpii_dataset.py(아직)



8. 가상환경에서 필요한거 다운

9. ln -s /usr/local/lib/python3.5/dist-packages/cv2.so ~/(가상환경이름)/lib/python3.5/site-packages/cv2.so
>> 이 방법으로 다운안되면 pip3 install opencv-python로 다운

10. mkdir src

11. cd src

12. cd deeppose

13. CUDA_VISIBLE_DEVICES=1 bash examples/train_lsp_alexnet_imagenet_small.sh
>>>원진 0번,  주영 1번, ...

14. 필요한거 다 깔기
(tensorflow설치>>)tensorflow version 1.4.1로 설치 >>pip3 install tensorflow-gpu==1.4.1

15. bash examples/train_lsp_alexnet_imagenet_small.sh 했을때 (SystemError: Parent module '' not loaded, cannot perform relative import)오류나면 

16. git pull 

17. 다시  bash examples/train_lsp_alexnet_imagenet_small.sh

18. 서버 내에서 training된 weight 가져오기 >>tar xvfz /tmp/out.tar.gz

19. restore하기

vi 파일이름.py  >>>>  

import tensorflow as tf

saver=tf.train.import_meta_graph("out/lsp_alexnet_imagenet_small/checkpoint-150000.meta",clear_devices=True)

with tf.Session() as sess:
   saver.restore(sess, "out/lsp_alexnet_imagenet_small/checkpoint-150000.data-00000-of-00001")

