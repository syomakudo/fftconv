#%%
import sys
import os
import argparse
from datasets import train_dataset,test_dataset,transform_simple,transform_to3ch
# from datasets import *
# from models import CNN_LayerModel, FFT_LayerModel
from models import CNNEx,FFT_CNNEx
from torch.utils.data import DataLoader
from torch import nn, optim
from trainer import train_cnn
from torchinfo import summary

def main(args):
    # transformer setting
    if args.dataset == 'mnist' or args.dataset == 'usps':
    # for mnist or usps
        src_transform = transform_simple(size=[args.size,args.size])
        trg_transform = transform_simple(size=[args.size,args.size])
        # src_transform = transform_to3ch(size=[args.size,args.size])
        # trg_transform = transform_to3ch(size=[args.size,args.size])
    else:
    # for svhn
        src_transform = transform_simple(size=[args.size,args.size])
        trg_transform = transform_simple(size=[args.size,args.size])

    # dataset setting
    source_dataset_train = train_dataset(args.dataset,src_transform)
    source_dataset_test = test_dataset(args.dataset_eval,trg_transform)

    # dataloader setting
    source_train_loader = DataLoader(
        source_dataset_train, args.batch_size, shuffle=True,drop_last=True,num_workers=2)
    source_test_loader = DataLoader(
        source_dataset_test, args.batch_size, shuffle=False,num_workers=2)
    
    model = None
    if args.arch == 'CNNEx':
        model = CNNEx(in_channels=args.in_channels,args=args).to(args.device)
    elif args.arch == 'FFT_CNNEx':
        model = FFT_CNNEx(in_channels=args.in_channels,args=args).to(args.device)

    summary(model=model,input_size=(1,args.in_channels,args.size,args.size))

    # loss setting    
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)

   #log
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    model = train_cnn(
        model, source_train_loader, source_test_loader,
        criterion, optimizer, args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=1) #fix 3
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--kernel_l1', type=int, default=111) #default=65, 129, 160
    parser.add_argument('--kernel_l2', type=int, default=65) #default=33
    parser.add_argument('--kernel_l3', type=int, default=33) #default=17
    parser.add_argument('--o_channels_l1', type=int, default=10) #default=20
    parser.add_argument('--o_channels_l2', type=int, default=10) #default=50
    parser.add_argument('--o_channels_l3', type=int, default=10) #default=50
    parser.add_argument('--learnnum_t', type=int, default=1000)
    parser.add_argument('--learnnum_v', type=int, default=1000)
    parser.add_argument('--size',type=int,default=256) #128
    parser.add_argument('--batch_size', type=int, default=8) #32
    parser.add_argument('--test', type=bool, default=False)
    # parser.add_argument('--arch', type=str, default='FFT_CNNEx') # CNNEx or FFT_CNNEx
    parser.add_argument('--arch', type=str, default='FFT_CNNEx')

    parser.add_argument('--comment', type=str, default='FFT_conv1')

    #option
    parser.add_argument('--kernel_l4', type=int, default=None)
    parser.add_argument('--kernel_l5', type=int, default=None) #使わないときはNone
    parser.add_argument('--o_channels_l4', type=int, default=None)
    parser.add_argument('--o_channels_l5', type=int, default=None)
    parser.add_argument('--padding_l1', type=int, default=0) #default=0
    parser.add_argument('--padding_l2', type=int, default=0) #default=0
    parser.add_argument('--padding_l3', type=int, default=0) #default=0
    parser.add_argument('--padding_l4', type=int, default=0) #default=0
    parser.add_argument('--padding_l5', type=int, default=0) #default=0
    
    # train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=30) #30
    # misc
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--message', '-m',  type=str, default='')

    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--dataset', type=str, default='mnist') #mnist
    parser.add_argument('--dataset_eval', type=str, default='mnist') # mnist or svhn or usps


    args, unknown = parser.parse_known_args()
    main(args)

    #[]: 学習時間が吾妻さんと違うことについて→それを踏まえてやってみる
    #[]: logファイルの深堀→csv化可能
    #ToDo: googleColoboratoryでやってみる 
    #ToDo: sizeを128より大きいサイズ1024とかにしてみる
    #ToDo: 各layerごとの順伝搬の速度、逆伝搬の速度をcnnと比較してみる。どこで差がついているのか
    #ToDo: 層を増やしてみる


    #Note: 出力チャネル数が少ないとFFTのほうがcnnより処理時間が短いが、出力チャネル数を増やすとfftのほうが遅くなる→原因不明
    #note: 大体の境界がtotalチャネル出力数60。これを超えるとfftのほうが処理時間高まる.
    
    #note: 理論的な計算量はガクッと落ちる。ただし実際の訓練時間が減るかどうかは、モデルが深くなることによるオーバーヘッドがあるので、フレームワークごとにまちまちで高速化しないこともある。

    #Note: 順伝搬のみのnnの計測時間を見るとtrainでかかる時間がvalidateでかかる時間の2倍かかる。cnnは両方同程度

    
    #Note: size:256、大カーネルでやるとfftのほうが断然早いが順伝搬速度はcnnのほうが10倍ほど早い
    '''
    cnn:
    0.0061958742141723635
    0.006498913764953613
    Epoch 1/30 | Train/Loss 1.692 Acc 0.392 | Val/Loss 0.957 Acc 73.219 Time 258.77s
    fft:
    0.05571802616119385
    0.055618197917938234
    Epoch 1/30 | Train/Loss 1.776 Acc 0.369 | Val/Loss 1.047 Acc 69.188 Time 19.01s
    '''
    
    #Todo: 出力のchannelを変えると処理時間も変わる。層を変えて調査していくときにchannelの総数は統一するべきか
    '''
    Dropoutに入るサイズを統一にしたほうがいい？→self.in_linearの値を統一するべきかという話
    パラメータを変えていくのにあたってどこを統一させないといけないかがはっきりしていない。
    今までは出力チャネル数の総数を統一させていた

    ・畳み込みが終了した時の出力画像サイズで統一
    ・self.in_linearで求めた値(全結合層に入る1次元テンソル)で統一させるのか
    感覚的には後者
    
    '''
    
    #note: やったこと
    '''

    '''
    
    #note: 吾妻さんの発表→fftで順伝搬の速度は変わらない。逆伝搬の方で違いが出ている説
    
    #Note: 聞いたことまとめ
    '''
    1epochあたりで出力されるTimeは全学習枚数にあたる順伝搬の時間の総数と逆伝搬の総数、パラメータ更新の全部の時間
    ・今は学習を含めた全部の時間を計測している
    →これからはfftconvだけさせてnn.leanerを入れないでただ畳み込みだけさせるコードを作って吾妻さんのように評価してみる
    →評価するに当たってcnnでAccが高く処理時間も短いパラメータを調べる。そのパラメータを固定させてfftとcnnを比較していく。カーネルサイズを変えたりする。
    ※このときに出力のチャネル数は大きくしなくても良い。
    ・カーネルのサイズは大きい方が効果良いらしいcnn。精度だかがよくなる。

    ・後々にmnist以外のデータセットでも調べていく。
    '''
    
    #Todo: cnnの好条件パラメータ探し
    #Todo: fft,cnn単体の畳み込みの時間の計測。順伝搬、逆伝搬。パラメータ固定fftの方が早い場合。
    #Todo: 学習を含めた全体の計測。
    
    
    #Note: 2022/12/10
    '''
    ・レイヤーごとの順伝搬をサイズ128で評価→cnnの方が早い kernelsizeの効果があまり発揮されてないように見える
    ・256でやるとfftの方が早く、cnnも遅くなる効果が確認できた
    展望
    もっと大きいサイズで試す
    '''
    
    #Note: 2022/12/11
    '''
    やったこと
    ・nn.Linearを含めた順伝搬、逆伝搬を各レイヤーごとに時間を計測。またニューラルネットワーク単位でも時間を計測した。
    ・nn.Linearを含めない順伝搬の時間を計測
    →256size以降の結果がfftの効果を確認することができた。
    →sizeによってfftも若干時間が遅くなっているが、これは入力のサイズが大きくなったからと考察。→カーネルのサイズを変えた時は時間はかわらないことを確認したい。
    展望
    ・nn.Linearなしの逆伝搬を計測
    →勾配はnn.Linearで計算されているため逆伝搬の処理が正しく動いているのかが心配。
    ・cnnの好条件探し
    ・カーネルサイズ変更して、精度を保ちつつ速度が早いfftの条件を探す
    
    '''
# %%
