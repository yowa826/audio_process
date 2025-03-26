Pythonにおける主要なFFTライブラリの比較
NumPy（numpy.fft）
主な特徴: NumPyはPython標準の数値計算ライブラリで、基本的なFFT機能を提供します。シンプルなAPIで使いやすく、追加の依存関係は不要です。numpy.fftモジュールでは高い精度（倍精度）の計算が行われ、float32入力は内部でfloat64に昇格して処理されます​
NUMPY.ORG
。そのため数値的な精度は良好ですが、メモリと時間面で単精度計算よりコストが増える場合があります。また、1次元から多次元までのFFT関数（fft, ifft, fft2, fftn など）が揃っています。柔軟性は高いものの、高度な最適化（マルチスレッド化やハードウェア固有最適化）は標準では有効になっていません。 設定可能なパラメータ: numpy.fftの関数ではFFTサイズ（n）や変換軸（axis）を指定できます。FFTサイズnを入力長より大きく指定すれば、自動的にゼロパディングされます​
DOCS.SCIPY.ORG
​
DOCS.SCIPY.ORG
（小さく指定すれば切り詰め）。多次元配列に対してはaxisやaxes引数でどの軸にFFTを適用するか選択可能です。窓関数の適用機能はライブラリ内にはなく、必要に応じてユーザがFFT前に信号に窓を掛ける必要があります。正規化についてはパラメータでの指定はできず、実装上は「forward（順方向）未正規化・inverse（逆変換）で1/N乗算」の方式が固定されています​
NUMPY.ORG
（※例えばifft実行時に1/Nが乗じられる）。 処理時間・パフォーマンス: NumPyのFFTはC言語実装（かつてはFFTPACK、現在はPocketFFT）に基づき、基本的にシングルスレッドで動作します​
GITHUB.COM
（インストール環境によってはIntel MKLによりマルチスレッド化される場合もありますが、標準では明示的制御不可）。一般的なサイズ（特に2のべき乗長）のFFTに対しては十分高速ですが、巨大なサイズや素数要素を含む長さでは高度に最適化されたFFTWなどに比べ遅くなる傾向があります​
BLOG.HPC.QMUL.AC.UK
。例えば、FFTWを使った実装（PyFFTW）は2のべき乗でないサイズのFFTでNumPy実装より大幅に高速になるケースがあります​
BLOG.HPC.QMUL.AC.UK
。NumPy自身には最適なアルゴリズムを選ぶ機能はなく、与えられた長さに対して決まった方法で計算します。とはいえ、高速フーリエ変換の計算量はO(N log N)であり、小～中規模のデータであれば多くの場合リアルタイム処理に足る性能を発揮します。 リアルタイム処理への適性: NumPy FFTはPythonから呼び出して都度計算するシンプルな方式のため、事前計画や状態の再利用といった仕組みはありません。連続したストリームに対して毎回numpy.fft.fftを呼ぶと、都度Python関数呼び出しのオーバーヘッドが発生します。しかし1回のFFT計算自体はC実装で高速に実行されるため、フレームサイズがせいぜい数千サンプル程度までのリアルタイム音声・信号処理であれば十分に低遅延で動作します。極端に短いフレームを高レートで処理する場合、Pythonレイヤのオーバーヘッドが無視できなくなるため、そのような用途では後述のPyFFTWのような手法（プラン再利用によるオーバーヘッド削減）が有利です。まとめると、NumPy FFTは手軽さと汎用性は高いものの、リアルタイム連続処理に特化した最適化はなされていません。 GPU対応: NumPyのFFTはGPU非対応です。計算は常にCPU上で行われ、GPUを直接利用することはできません。GPUを用いたFFT計算を行いたい場合、後述のCuPyなどGPU対応ライブラリを使う必要があります。 サンプルコード: 以下にNumPyを用いた1次元FFTの例を示します。長さ8の実数配列に対しFFTを計算し、結果（複素数列）を出力します。
python
コピーする
編集する
import numpy as np

# 長さ8のサンプル信号（実数）
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
X = np.fft.fft(x)  # 1次元FFTの計算

print(X)
# 出力例: [16.+0.j  -4.+4.j  0.+0.j  -4.-4.j   0.+0.j  -4.+4.j  0.+0.j  -4.-4.j]
SciPy（scipy.fft）
主な特徴: SciPyは科学技術計算向けの拡張ライブラリで、scipy.fftモジュールに高速フーリエ変換機能があります。SciPyのFFTはNumPyのFFTに対する上位互換となっており、APIは非常に似ていますが追加機能と最適化があります​
REALPYTHON.COM
​
REALPYTHON.COM
。たとえばマルチスレッド処理や高次の特殊変換（離散コサイン変換DCT、離散サイン変換DST、ハートレー変換など）にも対応しています​
GITHUB.COM
。SciPy 1.4以降の実装では内部でC++で書かれたPocketFFTライブラリを使用し、FFT計算時にSIMDベクトル命令を活用するなど高速化されています​
GITHUB.COM
。また、SciPyのFFTは単精度の計算もサポートしており（入力がfloat32ならcomplex64出力）、NumPyのような強制倍精度昇格は行いません。精度に関しては標準で倍精度演算ですが、必要に応じて単精度も扱える柔軟性があります。依存関係としては、SciPy自体がNumPyに依存しており、内部実装にFFTW等の外部ライブラリは不要です（PocketFFTはSciPyに同梱）。 指定できるパラメータ: 基本的なパラメータ指定はNumPyと同様です。FFT長さn、軸axis（または多次元FFT用にaxes）、および正規化モードnormを指定できます。特にnorm引数では"backward"（デフォルト）, "forward", "ortho"を指定可能で、順変換・逆変換どちらに1/N係数を適用するかや両方に1/√Nを適用するかを制御できます​
DOCS.SCIPY.ORG
。例えばnorm="forward"とすればFFT結果が1/Nで正規化され、逆FFT側ではスケーリングしない設定になります。窓関数についてもSciPyのFFT関数には組み込まれていませんが、SciPyにはscipy.signal.get_window関数で各種窓を生成する機能があるため、それを用いて入力信号に窓処理を施してからFFTに渡すことができます。さらにSciPy FFTの特徴として、workers引数で並列計算のワーカー数を指定できます​
DOCS.SCIPY.ORG
。例えばscipy.fft.fft(x, workers=4)のようにすると、データを分割して4並列でFFTを実行できる場合があります（※データが2次元以上で、複数の独立した1D変換に分けられる場合に有効​
DOCS.SCIPY.ORG
）。この機能により、複数信号のバッチFFTを高速化することが可能です。ゼロパディングや軸指定の挙動はNumPyと同一であり、SciPyにもfftshiftや周波数ビン計算用のfftfreq等の補助関数が用意されています。 処理時間・パフォーマンス: SciPyのFFTはより高度な最適化がなされており、NumPyのFFT実装と比べて高速になる場合があります。特に多次元配列に対するFFTや、非2べき長のFFTで性能差が報告されています​
GITHUB.COM
。SciPyのPocketFFTは、長さが素数要素を含むような場合にはBluesteinのアルゴリズムを用いて計算量オーダーを守る実装となっており、どんなサイズでもO(N log N)で効率を保ちます​
DOCS.SCIPY.ORG
。また、マルチスレッド対応のworkers引数を適切に使えば、同時に多数のFFTを計算する際にスループットが向上します。一方で単一の大きなFFTを1回計算する場合、基本的なアルゴリズム自体はNumPyと大差ないため、性能も同程度です。総じて、複数のFFTを扱う場面ではSciPyが有利であり、開発コミュニティも「可能ならSciPy実装を使うことが望ましい」としています​
REALPYTHON.COM
。例えばDask開発者も、SciPyのFFTは「ベクトル命令を活用し、オプションで共有メモリ並列もできるため、numpy.fftよりも多次元FFTでかなり高速」と報告しています​
GITHUB.COM
。 リアルタイム処理への適性: SciPy FFTも内部処理はC++実装で高速ですが、リアルタイム性に特化した機能は限定的です。workers引数による並列化は、一度に複数のFFTを計算する際のスループット向上には有用ですが、逐次的な単一FFTのレイテンシを下げるものではありません。それでも、SciPyにはscipy.fft.next_fast_len()という関数があり、与えられたサイズ以上でFFT計算に高速な長さ（通常は小さな素因数のみで構成される長さ）を計算できます​
DOCS.SCIPY.ORG
​
DOCS.SCIPY.ORG
。リアルタイム処理で速度がギリギリの場合、FFTサイズをこの関数で「次の高速長」にパディングすることで計算を高速化できる可能性があります。また、SciPyはPyFFTWなど他のFFT実装をバックエンドとして差し替える機構も持っており（SciPy v1.4以降）、scipy.fft.set_backend()を使ってPyFFTWを利用すればFFTWの計画最適化やスレッドを活用することもできます​
DOCS.CUPY.DEV
。したがって、工夫次第でリアルタイム性能を向上させることは可能ですが、標準のSciPy FFT自体にはプランの再利用（事前計画）機能はありません。リアルタイム性が極めて重要な場合、PyFFTWなどの活用が検討されますが、SciPy FFTも比較的低いオーバーヘッドで安定した性能を発揮する点で十分実用的です。 GPU対応: SciPyのFFTはGPUに直接対応していません（CPU計算のみ）。しかし、前述のようにSciPy 1.4+ではFFT計算のバックエンドを切り替える仕組みがあります。これを利用してCuPy（後述）をバックエンドに登録すれば、scipy.fft.fft呼び出しで実際にはGPU上の計算（cuFFT）が行われます​
DOCS.CUPY.DEV
​
DOCS.CUPY.DEV
。これはあくまで他のライブラリとの連携機能であり、SciPy単体でGPUを使うことはできません。したがって標準的にはGPU非対応ですが、間接的にGPU計算に繋げることは可能です。 サンプルコード: 以下はSciPyを用いた1次元FFTの例です（基本的な使い方はNumPyと同じです）。長さ16の実数配列にFFTを適用し、結果を表示します。
python
コピーする
編集する
import numpy as np
from scipy.fft import fft, fftfreq

# サンプル信号（正弦波に窓を適用した例）
fs = 1000  # サンプリング周波数
t = np.arange(0, 1, 1/fs)
signal = np.sin(2*np.pi*50*t) * np.hanning(len(t))  # 50Hzの正弦波 + ハン窓
X = fft(signal)  # FFT計算
freqs = fftfreq(len(signal), 1/fs)  # 周波数軸

# 結果の一部を表示
print(X[:5])
# 出力例: [ 0.000312+0.j -0.000314-0.000987j  0.000321-0.002088j ... ]
PyFFTW
主な特徴: PyFFTWは高速フーリエ変換ライブラリFFTWのPythonラッパーです​
PYFFTW.READTHEDOCS.IO
。FFTWはあらかじめ入念なプランニング（最適化計画）を行うことで非常に高速なFFTを実現するCライブラリであり、PyFFTWを使うことでその恩恵をPythonから得ることができます​
PYFFTW.READTHEDOCS.IO
。ユーザは必要な変換のサイズ・型を事前に指定してFFTW計画を作成でき、これによって繰り返し計算する場合に毎回最適化をやり直す必要がなくなります。PyFFTWはNumPyやSciPyのFFTインターフェースに準拠した関数も提供しており、既存コードの置き換えも容易です​
PYFFTW.READTHEDOCS.IO
。具体的にはpyfftw.interfaces.numpy_fftやpyfftw.interfaces.scipy_fftpackモジュールをインポートすると、それぞれのFFT関数をPyFFTW実装に差し替えて利用できます​
PYFFTW.READTHEDOCS.IO
。この互換インターフェースでは、例えばpyfftw.interfaces.numpy_fft.fft関数がnumpy.fft.fftと同等の使い勝手で動作します。また、PyFFTWにはメモリアラインメント確保のためのヘルパー（pyfftw.empty_alignedなど）があり、SIMD演算効率を高める工夫もされています​
PYFFTW.READTHEDOCS.IO
。依存関係としては、動作にFFTW3ライブラリが必要ですが、PyPI経由でインストールした場合はバイナリが付属していることもあります。 指定できるパラメータ: PyFFTWの低レベルAPI（pyfftw.FFTWクラス）では、FFT長や変換種類（FFT/IFFT）、データタイプ（単精度 complex64 or 倍精度 complex128）、変換軸、スレッド数、計画のプランニング方針（FFTW_MEASUREやFFTW_ESTIMATE等）を指定してFFTオブジェクトを構築します。これによりFFTWが最適なアルゴリズムを探索・計画します​
PYFFTW.READTHEDOCS.IO
。一方で互換インターフェースを使う場合、引数はNumPy/SciPyのFFT関数と同じです（例えばnやaxis引数など）。窓関数の扱いも基本的にライブラリ側では行わないため、必要なら入力データに乗じてからFFTします。PyFFTW特有の設定としては、グローバル設定でデフォルトのスレッド数やプランニング精度を指定可能です​
PYFFTW.READTHEDOCS.IO
​
PYFFTW.READTHEDOCS.IO
。例えばpyfftw.config.NUM_THREADSに整数を設定すると、インターフェース関数がFFTを実行する際のスレッド数を変更できます​
PYFFTW.READTHEDOCS.IO
（デフォルト1スレッド）。同様にpyfftw.config.PLANNER_EFFORTで'FFTW_ESTIMATE'や'FFTW_MEASURE'等を指定すれば、インターフェース関数利用時にもその設定でプランニングされます​
PYFFTW.READTHEDOCS.IO
。 処理時間・パフォーマンス: PyFFTW/FFTWは非常に高速で、特に繰り返し計算やサイズが非効率（素数長など）な場合に顕著な性能向上を示します。FFTWは初回に最適計画を練るため、最初の実行は少し時間がかかりますが、一度プランが構築されれば以降はそのプランを再利用して高速にFFTを計算します。例えば、PyFFTWの開発者によれば2のべき乗長ではNumPyと同程度かやや速い程度ですが、そうでないサイズではNumPyよりはるかに高速だったとの報告があります​
BLOG.HPC.QMUL.AC.UK
。実際、あるブログのベンチマークでは、サイズが2のべき乗でない大きな配列に対しPyFFTWはNumPyの数倍の速度を示しています​
BLOG.HPC.QMUL.AC.UK
。さらにPyFFTWはマルチコアを活用できます。NUM_THREADSを適切に設定すれば、複数スレッドで並列FFT計算を行い大きなサイズのFFTを高速化できます​
PYFFTW.READTHEDOCS.IO
​
PYFFTW.READTHEDOCS.IO
。これは特に大きな2D画像のFFTなどで有効です。加えて、PyFFTWにはプランのキャッシュ機構があり、一度計算したプランを一定時間保持して同じサイズ・型のFFTで再利用します​
BLOG.HPC.QMUL.AC.UK
。これによりスクリプト中で同サイズFFTを何度も呼び出す場合のオーバーヘッドが大幅に減少します。総じて、PyFFTWは単発のFFT計算を高速化するだけでなく、繰り返し実行時の効率も追求した実装と言えます。 リアルタイム処理への適性: PyFFTWはリアルタイム処理において非常に有用です。事前にFFTWプランを計画（例えばプランニングフラグにFFTW_MEASUREやFFTW_PATIENTを指定）しておけば、リアルタイムのループ内では既存プランによる計算を呼び出すだけになるため、呼び出しのオーバーヘッドと計算遅延を極小化できます​
BLOG.HPC.QMUL.AC.UK
。例えばオーディオストリームをフレームごとに処理する場合でも、PyFFTWなら初回にプランを作成しておき、各フレーム処理ではfftw_object()を呼ぶだけなので、ガベージコレクションや関数解決のオーバーヘッドが抑えられます。さらにスレッドを活かせば高スループットも維持できます。注意点として、最初のプラン計画には時間がかかるため（特に高精度なプランニングを選択した場合）、リアルタイムシステムでは起動時に余裕をもってプランを作成しておく必要があります。一度プランができればFFTWは非常に安定した低遅延動作を示すため、低遅延・高頻度のFFT処理にはPyFFTWが最適と言えます。 GPU対応: PyFFTW/FFTWはGPUには対応していません。FFTWはCPU向けに高度最適化されたライブラリであり、PyFFTWもそれをPythonから呼ぶものです。そのため計算はCPU上で行われ、GPUを利用したい場合はCuPyなど別のライブラリを併用する必要があります。 サンプルコード: PyFFTWを用いた1次元FFTの簡単な例を示します。まずPyFFTWインターフェースを有効化し、NumPy配列に対してFFTを計算しています。最後に通常のNumPy FFTと結果が同じであることを確認します。
python
コピーする
編集する
import numpy as np
import pyfftw
# PyFFTWの計画キャッシュを有効化（繰り返し呼び出しの効率化）
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4  # 任意: スレッド数を4に設定

# 長さ1024のランダムな実数信号
x = np.random.randn(1024)
# PyFFTW版FFTの計算（numpy_fftインターフェースを利用）
X_pyfftw = pyfftw.interfaces.numpy_fft.fft(x)
# NumPy版FFTの計算
X_np = np.fft.fft(x)

print(np.allclose(X_pyfftw, X_np))  # True（結果が一致）
※より高度な使い方として、pyfftw.FFTWクラスでFFTオブジェクトを作成し再利用する方法もありますが、ここでは簡潔さを優先してインターフェース経由の例を示しました。
CuPy
主な特徴: CuPyはNumPyとほぼ同等のAPIで使えるGPU対応ライブラリで、NVIDIA CUDA上で動作します。CuPyのcupy.fftモジュールではNumPyのFFT関数と同じインターフェースでGPU上のFFT計算が可能です​
DOCS.CUPY.DEV
。内部ではNVIDIAの提供するcuFFT（CUDA Fast Fourier Transform library）を呼び出しており、GPUハードウェア向けに最適化された高速なFFT計算が行われます​
DOCS.CUPY.DEV
。つまり、コード上はcp.fft.fft等と書くだけで、自動的にGPU上で大規模並列計算が行われる形になります。SciPyのscipy.fftに相当する機能もcupyx.scipy.fftとして一部提供されており​
DOCS.CUPY.DEV
、SciPyのバックエンドとしてCuPyを使うことも可能です。CuPy自体はCUDA対応GPUが必要であり、環境構築には対応するCUDA Toolkitに合わせたCuPyのインストールが必要です。精度に関しては、CuPyは基本的に単精度(32-bit)および倍精度(64-bit)に対応しています。さらに実験的ですが半精度(16-bit)FFTもサポートしており、対応ハードウェアでは単精度の約2倍速度で動作するとの報告があります​
DOCS.CUPY.DEV
。これはNumPyにはない特徴です（NumPyは半精度FFT未対応​
DOCS.CUPY.DEV
）。総じて、CuPyは大規模データの処理や機械学習分野でGPU計算をPythonから容易に利用するために設計されたライブラリであり、FFT計算もその一部として強力です。 指定できるパラメータ: CuPyのFFT関数（cupy.fft.fft, cupy.fft.ifftなど）は、基本的にNumPyのFFT関数と同じ引数を受け付けます。例えばcupy.fft.fft(x, n, axis, norm)のようにFFT長nや軸axis、正規化モードnormを指定できます（現時点でnorm引数が使えるかはバージョンによりますが、新しいCuPyではNumPy互換で実装されています）。窓関数も内部では扱いませんので、必要ならNumPy同様に事前に信号に掛けておきます。また、多次元FFTや実数FFT（rfft/irfft）も対応しています。加えてCuPyでは、CUDAのストリームを管理して非同期実行したり、バッチFFT（複数のFFTを一度に計算）を実行することも可能です。こうした高度な使い方には、cupy側でcufft.Plan1d等の低レベルAPIを用いる方法もありますが、一般的には高レベルなcupy.fft関数群で十分でしょう。 処理時間・パフォーマンス: 大規模なFFTにおいて、CuPy/cuFFTは驚異的な性能を発揮します。GPUは多数のコアと高帯域幅メモリを持つため、数百万～数千万ポイント級のFFTでもCPUを大きく上回るスループットで計算できます。ある比較では、サイズの大きい2D FFTにおいてcuFFTはPyFFTWより約10倍、NumPyより約100倍高速という結果も報告されています​
JOHNAPARKER.COM
。一方で、小さなサイズのFFTではGPUの起動・データ転送のオーバーヘッドが計算時間を上回り、CPUで直接計算した方が速い場合もあります​
STACKOVERFLOW.COM
​
STACKOVERFLOW.COM
。たとえば数百要素程度のFFTを1回だけ行うようなケースでは、データをGPUに送って結果を取り出す時間がもったいないため、NumPyの方が高速でしょう。従って、CuPyの真価は大きなデータサイズや多数回の繰り返し計算に現れます。そうした状況ではGPU並列性によるスピードアップがオーバーヘッドを凌駕し、トータルの計算時間を短縮できます。また、CuPyはGPUメモリ上で他の計算とFFTを組み合わせて実行できるため、機械学習のモデル内でスペクトル変換を行う場合などでもデータ移動を最小化して高効率に処理できます。 リアルタイム処理への適性: CuPyをリアルタイム用途に使う場合は、前提条件として計算対象のデータがGPU上に常駐していることが望ましいです。例えばセンサーデータが直接GPUに取得できる環境や、GPUで前段処理を行った結果に対してFFTを掛けるケースです。そうでない場合、毎フレームCPUからGPUへのデータ転送が発生し、これが遅延のボトルネックとなりがちです。フレームごとの処理時間を極限まで抑える必要があるリアルタイムシステムでは、GPUの並列計算能力を十分に活かせない可能性があります。しかし、もし処理すべきチャンネル数が非常に多い、またはフレームサイズ自体が大きいといった状況では、GPUでまとめてFFTを実行するメリットが出てきます。CuPy/cuFFTはバッチFFT（同じ長さのFFTを多数まとめて計算）に最適化されているため、例えば数百チャンネル分のFFTを同時実行するとCPUでは到底及ばないスループットを達成できます。また、GPU内部で完結する処理であれば遅延も安定して小さいです。したがって、低遅延かつ高スループットが要求され、かつデータ転送のオーバーヘッドを相殺できる規模であれば、CuPyはリアルタイム処理にも適性があります。CUDAストリームを駆使すれば他のGPU計算との並行実行も可能で、うまく設計すればCPUでは不可能な処理性能をリアルタイムシステムで引き出せるでしょう。 GPU対応: CuPyそのものがGPU専用ライブラリであり、GPU対応は完璧です。NVIDIA製のCUDA対応GPU上で動作し、FFTに関してはCUDAの公式FFT実装であるcuFFTを利用します。AMD等のGPUについてはCuPy自体は実験的にHIPなどをサポートしつつありますが、現時点では主にNVIDIA GPUを対象としています。CuPyを使う際には、適切なCUDA環境と対応するCuPyバージョンをインストールする必要があります。 サンプルコード: CuPyを用いた1次元FFTの例です。GPU上の配列に対してFFTを行い、結果をCPU側に取り出して確認しています。
python
コピーする
編集する
import cupy as cp

# GPU上のデータを用意（0～7の整数）
x_gpu = cp.arange(8, dtype=cp.float32)
X_gpu = cp.fft.fft(x_gpu)       # GPU上でFFT計算
X = cp.asnumpy(X_gpu)           # 計算結果をCPU上のnumpy配列に変換

print(X)
# 出力例: [28.+0.j -4.+9.656854j -4.+4.j -4.+1.656854j -4.+0.j -4.-1.656854j ...]
TensorFlow
主な特徴: TensorFlowはGoogleが開発した機械学習フレームワークですが、信号処理用にFFT関連のオペレーションも含んでいます。tf.signalモジュールにFFT機能があり、1次元から多次元までのFFTおよびIFFT、実数FFT（RFFT）などをサポートします。TensorFlowの利点は計算グラフによる最適化と自動微分機能で、FFTも計算グラフ上の演算として記述でき、必要なら勾配（逆伝搬）を計算することもできます。たとえば周波数領域での損失関数の勾配を求めるような応用では、フレームワークがFFT演算について自動的に微分を計算してくれます。TensorFlowはCPUとGPUの両方に対応しており、FFT演算も実行環境に応じて最適化されています。GPU上ではcuFFTを利用し、CPU上ではIntel MKLやEigen実装などで高速化されています（ビルド環境による）。TensorFlowは大規模データ処理や並列計算に強みがあり、FFTに関してもミニバッチで複数の信号を一度に処理したり、高次元の画像FFTを処理する用途に適します。ただし、ライブラリ自体が大規模なため、単純にFFTだけを目的とするにはオーバーヘッドが大きい点に注意が必要です。 指定できるパラメータ: TensorFlowのFFT関連関数は、tf.signal.fft（1次元FFT）、tf.signal.fft2d（2次元）、tf.signal.fft3d、およびそれぞれの逆変換（ifft系）、実数FFT用のtf.signal.rfft/irfftなどがあります。これらの関数ではFFT長を直接指定する引数はありませんが、実数FFTのrfftには代わりにfft_lengthという引数があり、これでゼロパディングや切り詰めを行うことができます​
KERAS.IO
（指定しない場合は入力長に等しいFFTを実施​
KERAS.IO
）。多次元FFTでは通常最後の軸もしくは特定の軸に対して変換を行う専用関数が用意されています（たとえばfft2dは最後の2軸に2D FFTを実施）。窓関数の適用はTensorFlow内部では行わないため、他のライブラリ同様にユーザがあらかじめ信号に窓を乗じる必要があります。正規化に関してTensorFlowでは、NumPyと同様の規約（逆変換で1/Nを掛ける）で固定されています。明示的に正規化を制御する引数はありません。例えばtf.signal.fftを行いtf.signal.ifftを適用すると元の配列に戻りますが、この際自動的に1/Nのスケーリングが行われています（ifft側で1/Nを乗じる実装）。その他、TensorFlowの演算なのでdtypeを指定することが重要です。入力型がtf.complex64やtf.complex128でない場合、自動的に実数として解釈されrfft系の結果になります。またtf.signal.fftに実数入力するとエラーになるため、実数しかない場合はtf.signal.rfftを使うか、tf.cast(x, tf.complex64)のように複素数型に変換してからfft関数を適用します。 処理時間・パフォーマンス: TensorFlowはGPU上で動作させた場合、cuFFTライブラリを通じて非常に高速にFFTを計算します​
STACKOVERFLOW.COM
。特にバッチサイズが大きい（同時に処理する信号本数が多い）場合や高次元のFFTでは、その並列処理能力のおかげでスループットが高いです。一方、単発の1D FFTをCPU上で実行するような場合、前後のグラフ構築やセッション実行のオーバーヘッドがあるため、単純なNumPy実装より遅くなることもあります。実際にTensorFlow 2系ではデフォルトでEager実行（逐次実行）になりましたが、それでも内部では最適化のための準備処理があり、FFT自体の計算時間は短くても全体の関数呼び出しに要する時間は純粋なC実装を直接呼ぶより長くなりがちです。従って、小規模なFFTではTensorFlowを使う利点は少なく、逆に大規模計算や学習モデルと組み合わせた場合に威力を発揮します。また、TensorFlowのデバイス間メモリ管理に乗るため、GPUとCPU間のデータ転送やメモリ確保のタイミングがフレームワークに一任され、効率良く隠蔽されます。総合的には、TensorFlowのFFTは大規模バッチ処理向きであり、単独のFFT性能だけで見れば専門ライブラリ（FFTW等）に劣る場合もありますが、大規模並列計算や自動微分の文脈では非常に有用です。 リアルタイム処理への適性: 一般的にTensorFlowは訓練や大規模バッチ推論向けに設計されているため、低レイテンシのリアルタイム処理にはあまり適していません。計算グラフを使う場合は初期化に時間がかかりますし、Eager実行でも内部のスケジューリングで遅延が生じることがあります。例えば音声ストリームをフレームごとに処理するのに逐一TensorFlowを呼び出すと、フレーム毎のオーバーヘッドが大きくなる可能性があります。もっとも、TensorFlowを用いた推論システムで前処理としてFFTが必要な場合など、リアルタイムシステム全体がTensorFlow上で構築されているケースでは、その一部としてFFTを使うのは合理的です。その場合でもTensorFlow Liteなどリアルタイム推論向けの軽量版を使うなどの工夫が必要でしょう。要するに、TensorFlowはリアルタイム低遅延専用ツールではないため、ミリ秒単位の応答が求められる用途では採用は慎重に検討すべきです。一方で、遅延よりスループット重視（例えば1秒間に大量のFFTをまとめて処理するような状況）であれば、GPUを最大限活用できるTensorFlowは有力な選択肢となります。 GPU対応: TensorFlowはGPUに完全対応しており、FFTもGPU上で実行可能です。TensorFlowをGPU環境で動かしている場合、tf.signal.fftなどを呼ぶと自動的に対応するGPU実装（cuFFT）が使われます​
STACKOVERFLOW.COM
。デフォルトの計算精度は複素64ビット（complex64）で、これは単精度の実部・虚部で構成される複素数です​
STACKOVERFLOW.COM
。複素128ビット（double precision）のFFTもCPU上では可能ですが、GPU上ではサポートが限定的で、TensorFlowの場合 GPUでcomplex128を扱えない（または非常に遅い）ため、自動的にCPUで計算されることがあります​
STACKOVERFLOW.COM
。実際、TensorFlowの古いバージョンではtf.fft（現tf.signal.fft）は入力をcomplex64に強制変換していました​
STACKOVERFLOW.COM
。したがってGPU利用時は基本的に単精度でのFFTになります。いずれにせよ、TensorFlowはCPU・GPUを意識せず同じコードで実行できますが、GPUメモリ上のデータを扱う際に最大性能を発揮する点は押さえておきましょう。 サンプルコード: TensorFlowを使ったFFTの例です。実数信号に対してtf.signal.rfftを用いてFFTを計算し、その結果をnumpy配列として取得します。
python
コピーする
編集する
import numpy as np
import tensorflow as tf

# TensorFlow用のデータ（実数のsin波）
t = np.linspace(0, 1, 1024, endpoint=False)
x = np.sin(2 * np.pi * 60 * t)  # 60Hzの正弦波
x_tensor = tf.constant(x, dtype=tf.float32)
X_tensor = tf.signal.rfft(x_tensor)     # 実数FFTの計算（複素出力）
X = X_tensor.numpy()                   # 結果をnumpy配列に変換

print(X.shape, X.dtype)
# 出力例: (513,) complex64
PyTorch
主な特徴: PyTorchはFacebook（現Meta）によって開発された深層学習ライブラリで、動的計算グラフとPythonライクな使いやすさが特徴です。PyTorchにもFFT機能が用意されており、torch.fftモジュール内でFFT/IFFTやrFFTなどが利用できます。NumPy的なインターフェースで、例えばtorch.fft.fftが1次元FFT、torch.fft.ifftがその逆変換になっています。PyTorchのテンソル演算はすべて自動微分に対応しており、FFTも同様に勾配計算が可能です（もっともFFTはユニタリではないので逆変換込みで差分をとるケースが多いでしょう）。PyTorchはTensorFlowと同様にCPUとGPUに対応し、演算はC++/CUDAで最適化されています。特筆すべきは動的な制御フローが可能な点で、Pythonの制御構造の中でテンソル計算を逐次行えるため、デバッグしやすくリアルタイム処理にも比較的組み込みやすいという利点があります。 指定できるパラメータ: PyTorchのFFT関数（torch.fft.fft, torch.fft.ifft等）はFFT長n、変換軸dim、正規化モードnormを指定できます​
FOSSIES.ORG
。normの指定肢や挙動はSciPy/NumPyと同様で、デフォルトが"backward"（逆変換で1/Nを掛ける）です​
FOSSIES.ORG
。PyTorchでは入力テンソルが実数の場合でもtorch.fft.fftをそのまま呼び出せ、内部で複素数に変換してからFFTが行われます​
FOSSIES.ORG
。このとき出力は全長の複素テンソルになり、エルミート対称なスペクトルが得られます（不要な負周波数成分も含む）。もし実数入力に対して一意な半分のスペクトルだけ欲しい場合はtorch.fft.rfftを使うことで、NumPyのrfftと同様に正の周波数成分のみの出力が得られます。窓関数も他のライブラリ同様に自前で掛ける必要があります。PyTorchではCPU上でもGPU上でも同じコードで動作しますが、テンソル自体をどちらのデバイスに置くか（.to('cuda')等で）を制御する形になります。 処理時間・パフォーマンス: PyTorchのFFTは、CPU上では高度に最適化されたライブラリ（例えばMKLやFFTWS）をバックエンドに使い、GPU上ではcuFFTを利用しています。そのため、性能はTensorFlowと同等か場合によっては勝ります。動的グラフゆえに余分な最適化ステップがなく、単発の演算でもオーバーヘッドが小さい傾向があります。PyTorch開発者によるドキュメントでも「CPU上では半精度（float16）および複素半精度（complex32）のFFTもサポート（ただし2のべき長に限る）」とあり​
FOSSIES.ORG
、最新のGPUではFP16による高速FFTも可能です。実際、PyTorchはcomplex64とcomplex128の両方をサポートしており、必要に応じて高精度計算も行えます。もっとも、複素128でのGPU計算は非常に遅いため、GPU使用時はcomplex64が主になります。総じて、PyTorch FFTは深層学習モデル内で使用するFFTとして最適化されており、大規模なバッチFFTやGPU演算との組み合わせで高い性能を発揮します。一方、FFT単体のベンチマークでは、PyFFTWなどに比べると若干劣ることも考えられますが、それでもNumPy標準実装よりはるかに高速です。特にGPU上の大きなFFTでは、TensorFlow同様に大きな利点があります。 リアルタイム処理への適性: PyTorchは動的に計算を進められるため、リアルタイム処理への組み込みもしやすいと言えます。例えば音声ストリーム処理で逐次フレームにPyTorchのテンソル変換を適用し、FFTを計算してスペクトルをPyTorch上で畳み込み、その結果を逆FFTして…といった処理を、Pythonのforループ内で直接書くことができます。TensorFlowではこのような逐次処理はEagerモードでないと難しく、Eagerで行うと却ってオーバーヘッドが大きいことがありますが、PyTorchは元来Eager相当なので問題ありません。とはいえ、PyTorch自体は主にGPU資源を活用する大規模処理に焦点があるため、極端に厳密なリアルタイム制約（例: 1ミリ秒以内の処理）においてはオーバーヘッドがゼロではない点に注意が必要です。モデルをまたいだ遅延なども考慮すると、PyTorchだけで完結しない部分とのインターフェースで遅延が発生し得ます。しかし、一般に数十ミリ秒程度のフレーム処理であれば十分リアルタイムに動作可能であり、Python＋PyTorchだけで音声処理システムを構築している事例もあります。要するに、PyTorch FFTはリアルタイム処理とオフライン処理の両面に適したバランスを持っていると言えるでしょう。 GPU対応: PyTorchはGPUに対応しています。CPU上のテンソルに対してtorch.fft.fftを呼べばCPUで計算され、GPU上のテンソル（device='cuda'に移したもの）に対して呼べば自動的にcuFFTを利用したGPU計算が行われます。ユーザはデータをどのデバイスに置くかだけ意識すればよく、FFT関数自体は共通です。また、PyTorchはマルチGPUもサポートしており、DistributedDataParallelなどを組み合わせれば複数GPUで並行してFFTを含む計算をさせることもできます。ただし単体のFFTを複数GPUで分割計算するようなことは通常せず、あくまでバッチをGPU間で分散させる使い方になります。 サンプルコード: PyTorchで1次元FFTを実行する例です。実数テンソルを用意しFFTを計算、結果のテンソルを表示します。
python
コピーする
編集する
import torch

# PyTorchテンソル（実数）
x = torch.tensor([0.0, 1.0, 0.0, -1.0])
X = torch.fft.fft(x)  # 1次元FFT

print(X)
# 出力例: tensor([0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j])
Numba（JIT最適化）
主な特徴: NumbaはライブラリというよりPythonコードを高速化するためのJust-In-Timeコンパイラです。Numbaを使うと、Pythonで書いた数値計算のループ処理等を機械語にコンパイルして高速に実行できます。FFT専用の関数は提供していませんが、ユーザがFFTアルゴリズムを実装した場合にその計算を高速化したり、複数のFFT処理を並列化したりすることが可能です。たとえばPythonで単純なO(N^2)のDFTを実装すると通常は非常に遅いですが、Numbaの@njitデコレータを付ければC言語並みの速度で実行されます。また、NumbaはNumPy配列を扱えるため、FFTの前後処理（窓関数適用やフィルタ演算など）をまとめてJITコンパイルして効率化する、といった使い方も考えられます。依存関係はNumba本体（LLVMベース）とNumPyのみで、軽量です。 指定できるパラメータ: Numba自体にはFFTの機能がないため、パラメータはユーザ実装次第です。たとえば自前でコoley–TukeyのFFTを実装すれば、再帰的にサイズを処理するような柔軟性を盛り込むこともできますし、固定サイズ専用に書いてしまうこともできます。NumbaでJIT化した関数は、コンパイル時にデータ型や配列サイズが決定され最適化されます（可変長にも対応しますが、最初の呼び出し時にコンパイルが走る仕組みです）。一般に、FFT長や軸といった情報はユーザが関数内で扱うことになり、Numbaがそれを解釈するわけではありません。また、NumbaのJIT関数内からライブラリ関数を呼ぶことも制限があります。残念ながらNumPyやSciPyのFFT関数はNumbaのサポートリストに含まれておらず、そのままではnjitコンパイルできません​
STACKOVERFLOW.COM
。そのため、NumbaでFFTをやりたい場合は自前でアルゴリズムを記述する必要があります。一部、有志が作成したNumba対応のFFT実装（例：Rocket-FFT​
GITHUB.COM
）もありますが、ここでは一般論として説明します。 処理時間・パフォーマンス: Numbaで最適化したFFT相当コードの性能は、実装内容とデータサイズに強く依存します。自前で最適なFFTアルゴリズム（O(N log N)）を実装できたとしても、それがFFTWやcuFFTほどチューニングされている可能性は低く、純粋な速度では劣るでしょう。しかし、Pythonで実装した単純なアルゴリズムでも、NumbaでJITコンパイルすればPythonインタプリタのオーバーヘッドを排除できるため、数十倍以上の速度向上が見込めます。また、Numbaはparallel=Trueオプションでデータ並列化も可能です。例えば「多数の小さなFFTを複数スレッドで同時計算する」ような並列処理をユーザが書けば、Numbaがスレッド並列にコンパイルしてくれます。ただし前述の通り、内部でnp.fftなどを呼ぶのはサポート外であり​
STACKOVERFLOW.COM
、純粋Pythonのループ処理部分のみが並列化対象になります。つまり、高速なライブラリ呼び出し部分は並列化できない点に注意が必要です。一方で、NumbaはGPU向けのコード（CUDAカーネル）を書くこともできます。CUDA Pythonを用いればGPU上で実行されるカスタムFFTカーネルを記述することも理論上可能です。しかし、FFTのように複雑でメモリアクセスパターンの重要なアルゴリズムをGPUで一から実装するのは非常に難しく、性能チューニングも容易ではありません。現実的には、GPUに関してはCuPyやカスタムCUDA Cコードに任せ、NumbaはCPU上での最適化に用いるのが一般的です。総合すると、Numbaを使えば「ライブラリと純Pythonコードの中間」のような性能が得られ、特定のニッチな要求に合わせて最適化を試せる利点があります。 リアルタイム処理への適性: NumbaでJIT化したコードは、うまく書けばC/C++で書いたのと同等の速度になります。そのため、リアルタイム処理のボトルネックがPythonの遅さにある場合、Numba導入は極めて有効です。例えば毎フレームごとに特殊なスペクトル操作を行うような処理をPythonだけで記述していたケースでは、Numbaでコンパイルするだけで処理落ちしなくなる、といったことがあります。FFTそのものについて言えば、前述のようにNumbaでライブラリ並みの高速化をするのはハードルが高いですが、「FFT + α」の処理を一体化してコンパイルすることによりデータのやり取りや関数呼び出し回数を減らし、結果的に遅延を縮めることは可能です。例えば窓掛け・FFT・周波数ドメインフィルタ・逆FFTまでを一つのNumba関数で実装すれば、途中でPythonに戻らない分だけ低遅延になります。また、ガベージコレクションの発生も抑えられ安定した実行時間が得られるでしょう。リアルタイム処理では一度コンパイルしてしまえば繰り返し高速に動作するNumbaの特性は好適です。留意点として、Numba関数の初回実行時にはコンパイルが走るため僅かな遅延があります（事前にダミー呼び出ししてウォームアップ可能）。さらに、Numbaでの最適化の度合いはコードの書き方によって大きく変わります。メモリアクセスのパターンやループの使い方次第では最適化しきれない場合もあるため、プロファイリングしながら調整が必要です。以上より、NumbaはリアルタイムDSPの自作実装を後押しする強力なツールですが、その効果は開発者の実装力に左右される部分もあります。 GPU対応: 前述のとおり、NumbaにはGPU向けコンパイル機能（CUDA Python）があり、カスタムCUDAカーネルを書くことでGPUを利用可能です。ただし、高品質なFFTカーネルを自作するのは難易度が高く、現実的には既存のcuFFTを使う方が賢明です。Numbaが得意とするのは、GPU計算で不足する部分を補完する軽量なカーネルを書くことです。例えばデータの転置処理や簡単な並列フィルタリングなどは自作カーネルで対応し、肝心のFFT計算自体はCuPy/cuFFTに任せる、といった使い分けが考えられます。2025年現在、Numba本体にGPU向けFFT計算を簡単に行うためのユーティリティは提供されていません。したがって、Numba単体ではFFTのGPU実行はすぐには実現できないというのが実情です（自力でCUDAカーネルを書く場合は別ですが）。まとめると、Numbaは主にCPU上での最適化に向いており、GPUでFFTをしたい場合は他ライブラリとの併用を検討すべきです。 サンプルコード: Numbaを使って直接DFTを実装し、高速化する例を示します。長さNの複素離散信号に対し、直接計算でDFTを求める関数を@njitでJITコンパイルしています（アルゴリズムはO(N^2)でありFFTではありませんが、動作確認目的の簡単な例です）。
python
コピーする
編集する
import numpy as np
from numba import njit

@njit
def dft(x):
    N = x.shape[0]
    X = np.empty(N, dtype=np.complex128)
    for k in range(N):
        s = 0+0j
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s
    return X

# サンプル信号（複素数列）
x = np.array([1+0j, 2+1j, 0-1j, -1+2j])
print(dft(x))
# 出力例（numpy.fft.fftと同じ結果）: [2.+2.j 3.-0.j 0.-4.j 1.-0.j]
上記のdft関数はNumbaによってコンパイルされているため、Pythonでそのまま二重ループを回すよりも格段に高速に動作します（Nが大きくなるほど差が開きます）。このようにNumbaを使えば、自作アルゴリズムであってもコンパイル最適化によってスピードアップでき、用途に合わせた柔軟な実装が可能です。
ライブラリの機能・性能比較
各ライブラリの特徴をまとめた比較表を示します。

了解しました。PythonでFFT処理を行うための主要ライブラリについて、それぞれの特徴、指定可能なパラメータ、処理速度、リアルタイム処理の適性、GPU対応の有無などを日本語で詳しくまとめます。あわせて、各ライブラリごとのFFT処理のサンプルコードも用意し、最後に全体を比較した要約表を掲載します。

# Pythonにおける主要なFFTライブラリの比較

## NumPy（numpy.fft）
**主な特徴:** NumPyはPython標準の数値計算ライブラリで、基本的なFFT機能を提供します。シンプルなAPIで使いやすく、追加の依存関係は不要です。`numpy.fft`モジュールでは高い精度（倍精度）の計算が行われ、`float32`入力は内部で`float64`に昇格して処理されます ([Discrete Fourier Transform (numpy.fft) — NumPy v2.2 Manual](https://numpy.org/doc/stable/reference/routines.fft.html#:~:text=Type%20Promotion))。そのため数値的な精度は良好ですが、メモリと時間面で単精度計算よりコストが増える場合があります。また、1次元から多次元までのFFT関数（fft, ifft, fft2, fftn など）が揃っています。柔軟性は高いものの、高度な最適化（マルチスレッド化やハードウェア固有最適化）は標準では有効になっていません。

**設定可能なパラメータ:** `numpy.fft`の関数では**FFTサイズ**（`n`）や**変換軸**（`axis`）を指定できます。FFTサイズ`n`を入力長より大きく指定すれば、自動的にゼロパディングされます ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=n%20int%2C%20optional)) ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=Length%20of%20the%20transformed%20axis,specified%20by%20axis%20is%20used))（小さく指定すれば切り詰め）。多次元配列に対しては`axis`や`axes`引数でどの軸にFFTを適用するか選択可能です。**窓関数**の適用機能はライブラリ内にはなく、必要に応じてユーザがFFT前に信号に窓を掛ける必要があります。**正規化**についてはパラメータでの指定はできず、実装上は「forward（順方向）未正規化・inverse（逆変換）で1/N乗算」の方式が固定されています ([Discrete Fourier Transform (numpy.fft) — NumPy v2.2 Manual](https://numpy.org/doc/stable/reference/routines.fft.html#:~:text=numpy.fft%20promotes%20,fftpack))（※例えば`ifft`実行時に1/Nが乗じられる）。

**処理時間・パフォーマンス:** NumPyのFFTはC言語実装（かつてはFFTPACK、現在はPocketFFT）に基づき、基本的に**シングルスレッド**で動作します ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))（インストール環境によってはIntel MKLによりマルチスレッド化される場合もありますが、標準では明示的制御不可）。一般的なサイズ（特に2のべき乗長）のFFTに対しては十分高速ですが、巨大なサイズや素数要素を含む長さでは高度に最適化されたFFTWなどに比べ遅くなる傾向があります ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))。例えば、FFTWを使った実装（PyFFTW）は2のべき乗でないサイズのFFTでNumPy実装より大幅に高速になるケースがあります ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))。NumPy自身には最適なアルゴリズムを選ぶ機能はなく、与えられた長さに対して決まった方法で計算します。とはいえ、高速フーリエ変換の計算量はO(N log N)であり、小～中規模のデータであれば多くの場合リアルタイム処理に足る性能を発揮します。

**リアルタイム処理への適性:** NumPy FFTはPythonから呼び出して都度計算するシンプルな方式のため、**事前計画**や**状態の再利用**といった仕組みはありません。連続したストリームに対して毎回`numpy.fft.fft`を呼ぶと、都度Python関数呼び出しのオーバーヘッドが発生します。しかし1回のFFT計算自体はC実装で高速に実行されるため、フレームサイズがせいぜい数千サンプル程度までのリアルタイム音声・信号処理であれば十分に低遅延で動作します。極端に短いフレームを高レートで処理する場合、Pythonレイヤのオーバーヘッドが無視できなくなるため、そのような用途では後述のPyFFTWのような手法（プラン再利用によるオーバーヘッド削減）が有利です。まとめると、NumPy FFTは**手軽さと汎用性**は高いものの、リアルタイム連続処理に特化した最適化はなされていません。

**GPU対応:** NumPyのFFTは**GPU非対応**です。計算は常にCPU上で行われ、GPUを直接利用することはできません。GPUを用いたFFT計算を行いたい場合、後述のCuPyなどGPU対応ライブラリを使う必要があります。

**サンプルコード:** 以下にNumPyを用いた1次元FFTの例を示します。長さ8の実数配列に対しFFTを計算し、結果（複素数列）を出力します。

```python
import numpy as np

# 長さ8のサンプル信号（実数）
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
X = np.fft.fft(x)  # 1次元FFTの計算

print(X)
# 出力例: [16.+0.j  -4.+4.j  0.+0.j  -4.-4.j   0.+0.j  -4.+4.j  0.+0.j  -4.-4.j]
```

## SciPy（scipy.fft）
**主な特徴:** SciPyは科学技術計算向けの拡張ライブラリで、`scipy.fft`モジュールに高速フーリエ変換機能があります。SciPyのFFTはNumPyのFFTに対する**上位互換**となっており、APIは非常に似ていますが追加機能と最適化があります ([Fourier Transforms With scipy.fft: Python Signal Processing – Real Python](https://realpython.com/python-scipy-fft/#:~:text=,instead)) ([Fourier Transforms With scipy.fft: Python Signal Processing – Real Python](https://realpython.com/python-scipy-fft/#:~:text=contains%20more%20features%20and%20is,should%20use%20the%20SciPy%20implementation))。たとえば**マルチスレッド処理**や**高次の特殊変換**（離散コサイン変換DCT、離散サイン変換DST、ハートレー変換など）にも対応しています ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=,precision%20transforms))。SciPy 1.4以降の実装では内部でC++で書かれたPocketFFTライブラリを使用し、FFT計算時にSIMDベクトル命令を活用するなど高速化されています ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))。また、SciPyのFFTは**単精度**の計算もサポートしており（入力がfloat32ならcomplex64出力）、NumPyのような強制倍精度昇格は行いません。精度に関しては標準で倍精度演算ですが、必要に応じて単精度も扱える柔軟性があります。依存関係としては、SciPy自体がNumPyに依存しており、内部実装にFFTW等の外部ライブラリは不要です（PocketFFTはSciPyに同梱）。

**指定できるパラメータ:** 基本的なパラメータ指定はNumPyと同様です。**FFT長さ**`n`、**軸**`axis`（または多次元FFT用に`axes`）、および**正規化モード**`norm`を指定できます。特に`norm`引数では`"backward"`（デフォルト）, `"forward"`, `"ortho"`を指定可能で、順変換・逆変換どちらに1/N係数を適用するかや両方に1/√Nを適用するかを制御できます ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=norm))。例えば`norm="forward"`とすればFFT結果が1/Nで正規化され、逆FFT側ではスケーリングしない設定になります。**窓関数**についてもSciPyのFFT関数には組み込まれていませんが、SciPyには`scipy.signal.get_window`関数で各種窓を生成する機能があるため、それを用いて入力信号に窓処理を施してからFFTに渡すことができます。さらにSciPy FFTの特徴として、**workers**引数で並列計算のワーカー数を指定できます ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=workers%20int%2C%20optional))。例えば`scipy.fft.fft(x, workers=4)`のようにすると、データを分割して4並列でFFTを実行できる場合があります（※データが2次元以上で、複数の独立した1D変換に分けられる場合に有効 ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=The%20,may%20be%20used%20than%20requested))）。この機能により、複数信号のバッチFFTを高速化することが可能です。ゼロパディングや軸指定の挙動はNumPyと同一であり、SciPyにも`fftshift`や周波数ビン計算用の`fftfreq`等の補助関数が用意されています。

**処理時間・パフォーマンス:** SciPyのFFTは**より高度な最適化**がなされており、NumPyのFFT実装と比べて高速になる場合があります。特に多次元配列に対するFFTや、非2べき長のFFTで性能差が報告されています ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))。SciPyのPocketFFTは、長さが素数要素を含むような場合にはBluesteinのアルゴリズムを用いて計算量オーダーを守る実装となっており、どんなサイズでもO(N log N)で効率を保ちます ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=FFT%20,padding%20the%20input%20using%20next_fast_len))。また、マルチスレッド対応の`workers`引数を適切に使えば、同時に多数のFFTを計算する際にスループットが向上します。一方で単一の大きなFFTを1回計算する場合、基本的なアルゴリズム自体はNumPyと大差ないため、性能も同程度です。総じて、**複数のFFTを扱う場面ではSciPyが有利**であり、開発コミュニティも「可能ならSciPy実装を使うことが望ましい」としています ([Fourier Transforms With scipy.fft: Python Signal Processing – Real Python](https://realpython.com/python-scipy-fft/#:~:text=contains%20more%20features%20and%20is,should%20use%20the%20SciPy%20implementation))。例えばDask開発者も、SciPyのFFTは「ベクトル命令を活用し、オプションで共有メモリ並列もできるため、numpy.fftよりも多次元FFTでかなり高速」と報告しています ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))。

**リアルタイム処理への適性:** SciPy FFTも内部処理はC++実装で高速ですが、**リアルタイム性**に特化した機能は限定的です。`workers`引数による並列化は、一度に複数のFFTを計算する際のスループット向上には有用ですが、逐次的な単一FFTのレイテンシを下げるものではありません。それでも、SciPyには`scipy.fft.next_fast_len()`という関数があり、与えられたサイズ以上でFFT計算に高速な長さ（通常は小さな素因数のみで構成される長さ）を計算できます ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=FFT%20,padding%20the%20input%20using%20next_fast_len)) ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=FFT%20,padding%20the%20input%20using%20next_fast_len))。リアルタイム処理で速度がギリギリの場合、FFTサイズをこの関数で「次の高速長」にパディングすることで計算を高速化できる可能性があります。また、SciPyはPyFFTWなど他のFFT実装を**バックエンドとして差し替える**機構も持っており（SciPy v1.4以降）、`scipy.fft.set_backend()`を使ってPyFFTWを利用すればFFTWの計画最適化やスレッドを活用することもできます ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=Since%20SciPy%20v1,set_backend%28%29%20can%20be%20used))。したがって、工夫次第でリアルタイム性能を向上させることは可能ですが、**標準のSciPy FFT自体にはプランの再利用**（事前計画）**機能はありません**。リアルタイム性が極めて重要な場合、PyFFTWなどの活用が検討されますが、SciPy FFTも比較的低いオーバーヘッドで安定した性能を発揮する点で十分実用的です。

**GPU対応:** SciPyのFFTは**GPUに直接対応していません**（CPU計算のみ）。しかし、前述のようにSciPy 1.4+ではFFT計算のバックエンドを切り替える仕組みがあります。これを利用してCuPy（後述）をバックエンドに登録すれば、`scipy.fft.fft`呼び出しで実際にはGPU上の計算（cuFFT）が行われます ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=Since%20SciPy%20v1,set_backend%28%29%20can%20be%20used)) ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=import%20cupy%20as%20cp%20import,fft))。これはあくまで他のライブラリとの連携機能であり、SciPy単体でGPUを使うことはできません。したがって標準的には**GPU非対応**ですが、間接的にGPU計算に繋げることは可能です。

**サンプルコード:** 以下はSciPyを用いた1次元FFTの例です（基本的な使い方はNumPyと同じです）。長さ16の実数配列にFFTを適用し、結果を表示します。

```python
import numpy as np
from scipy.fft import fft, fftfreq

# サンプル信号（正弦波に窓を適用した例）
fs = 1000  # サンプリング周波数
t = np.arange(0, 1, 1/fs)
signal = np.sin(2*np.pi*50*t) * np.hanning(len(t))  # 50Hzの正弦波 + ハン窓
X = fft(signal)  # FFT計算
freqs = fftfreq(len(signal), 1/fs)  # 周波数軸

# 結果の一部を表示
print(X[:5])
# 出力例: [ 0.000312+0.j -0.000314-0.000987j  0.000321-0.002088j ... ]
```

## PyFFTW
**主な特徴:** PyFFTWは高速フーリエ変換ライブラリFFTWのPythonラッパーです ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=FFTW%20is%20a%20very%20fast,fastest%20way%2C%20so%20called%20planning))。FFTWはあらかじめ入念な**プランニング（最適化計画）**を行うことで非常に高速なFFTを実現するCライブラリであり、PyFFTWを使うことでその恩恵をPythonから得ることができます ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=FFTW%20is%20a%20very%20fast,fastest%20way%2C%20so%20called%20planning))。ユーザは**必要な変換のサイズ・型**を事前に指定してFFTW計画を作成でき、これによって繰り返し計算する場合に毎回最適化をやり直す必要がなくなります。PyFFTWはNumPyやSciPyのFFTインターフェースに準拠した関数も提供しており、既存コードの置き換えも容易です ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=Quick%20and%20easy%3A%20the%20pyfftw,module%2012))。具体的には`pyfftw.interfaces.numpy_fft`や`pyfftw.interfaces.scipy_fftpack`モジュールをインポートすると、それぞれのFFT関数をPyFFTW実装に差し替えて利用できます ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=Quick%20and%20easy%3A%20the%20pyfftw,module%2012))。この互換インターフェースでは、例えば`pyfftw.interfaces.numpy_fft.fft`関数が`numpy.fft.fft`と同等の使い勝手で動作します。また、PyFFTWには**メモリアラインメント**確保のためのヘルパー（`pyfftw.empty_aligned`など）があり、SIMD演算効率を高める工夫もされています ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=We%20initially%20create%20and%20fill,exists%20to%20align%20a%20pre))。依存関係としては、動作にFFTW3ライブラリが必要ですが、PyPI経由でインストールした場合はバイナリが付属していることもあります。

**指定できるパラメータ:** PyFFTWの低レベルAPI（`pyfftw.FFTW`クラス）では、FFT長や変換種類（FFT/IFFT）、**データタイプ**（単精度 complex64 or 倍精度 complex128）、**変換軸**、**スレッド数**、**計画のプランニング方針**（FFTW_MEASUREやFFTW_ESTIMATE等）を指定してFFTオブジェクトを構築します。これによりFFTWが最適なアルゴリズムを探索・計画します ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=FFTW%20is%20a%20very%20fast,fastest%20way%2C%20so%20called%20planning))。一方で互換インターフェースを使う場合、引数はNumPy/SciPyのFFT関数と同じです（例えば`n`や`axis`引数など）。窓関数の扱いも基本的にライブラリ側では行わないため、必要なら入力データに乗じてからFFTします。PyFFTW特有の設定としては、グローバル設定で**デフォルトのスレッド数**や**プランニング精度**を指定可能です ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=interfaces%20,code%20demonstrates%20using%20the%20pyfftw)) ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=For%20example%2C%20to%20change%20the,and%20specify%204%20threads))。例えば`pyfftw.config.NUM_THREADS`に整数を設定すると、インターフェース関数がFFTを実行する際のスレッド数を変更できます ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=interfaces%20,code%20demonstrates%20using%20the%20pyfftw))（デフォルト1スレッド）。同様に`pyfftw.config.PLANNER_EFFORT`で`'FFTW_ESTIMATE'`や`'FFTW_MEASURE'`等を指定すれば、インターフェース関数利用時にもその設定でプランニングされます ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=For%20example%2C%20to%20change%20the,and%20specify%204%20threads))。

**処理時間・パフォーマンス:** PyFFTW/FFTWは**非常に高速**で、特に**繰り返し計算**や**サイズが非効率（素数長など）**な場合に顕著な性能向上を示します。FFTWは初回に最適計画を練るため、最初の実行は少し時間がかかりますが、一度プランが構築されれば以降はそのプランを再利用して高速にFFTを計算します。例えば、PyFFTWの開発者によれば**2のべき乗長ではNumPyと同程度かやや速い程度ですが、そうでないサイズではNumPyよりはるかに高速**だったとの報告があります ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))。実際、あるブログのベンチマークでは、サイズが2のべき乗でない大きな配列に対しPyFFTWはNumPyの数倍の速度を示しています ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))。さらにPyFFTWは**マルチコア**を活用できます。NUM_THREADSを適切に設定すれば、複数スレッドで並列FFT計算を行い大きなサイズのFFTを高速化できます ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=,threaded%29%20pyfftw.config.NUM_THREADS%20%3D%20multiprocessing.cpu_count)) ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=match%20at%20L442%20pyfftw,4))。これは特に大きな2D画像のFFTなどで有効です。加えて、PyFFTWには**プランのキャッシュ**機構があり、一度計算したプランを一定時間保持して同じサイズ・型のFFTで再利用します ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=After%20re,type%20and%20size%20are%20reused))。これによりスクリプト中で同サイズFFTを何度も呼び出す場合のオーバーヘッドが大幅に減少します。総じて、PyFFTWは**単発のFFT計算を高速化するだけでなく、繰り返し実行時の効率も追求**した実装と言えます。

**リアルタイム処理への適性:** PyFFTWはリアルタイム処理において非常に有用です。事前にFFTWプランを計画（例えばプランニングフラグにFFTW_MEASUREやFFTW_PATIENTを指定）しておけば、リアルタイムのループ内では既存プランによる計算を呼び出すだけになるため、**呼び出しのオーバーヘッドと計算遅延を極小化**できます ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=After%20re,type%20and%20size%20are%20reused))。例えばオーディオストリームをフレームごとに処理する場合でも、PyFFTWなら初回にプランを作成しておき、各フレーム処理では`fftw_object()`を呼ぶだけなので、ガベージコレクションや関数解決のオーバーヘッドが抑えられます。さらにスレッドを活かせば高スループットも維持できます。注意点として、最初のプラン計画には時間がかかるため（特に高精度なプランニングを選択した場合）、リアルタイムシステムでは起動時に余裕をもってプランを作成しておく必要があります。一度プランができればFFTWは非常に安定した低遅延動作を示すため、**低遅延・高頻度のFFT処理にはPyFFTWが最適**と言えます。

**GPU対応:** PyFFTW/FFTWは**GPUには対応していません**。FFTWはCPU向けに高度最適化されたライブラリであり、PyFFTWもそれをPythonから呼ぶものです。そのため計算はCPU上で行われ、GPUを利用したい場合はCuPyなど別のライブラリを併用する必要があります。

**サンプルコード:** PyFFTWを用いた1次元FFTの簡単な例を示します。まずPyFFTWインターフェースを有効化し、NumPy配列に対してFFTを計算しています。最後に通常のNumPy FFTと結果が同じであることを確認します。

```python
import numpy as np
import pyfftw
# PyFFTWの計画キャッシュを有効化（繰り返し呼び出しの効率化）
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4  # 任意: スレッド数を4に設定

# 長さ1024のランダムな実数信号
x = np.random.randn(1024)
# PyFFTW版FFTの計算（numpy_fftインターフェースを利用）
X_pyfftw = pyfftw.interfaces.numpy_fft.fft(x)
# NumPy版FFTの計算
X_np = np.fft.fft(x)

print(np.allclose(X_pyfftw, X_np))  # True（結果が一致）
```

※より高度な使い方として、`pyfftw.FFTW`クラスでFFTオブジェクトを作成し再利用する方法もありますが、ここでは簡潔さを優先してインターフェース経由の例を示しました。

## CuPy
**主な特徴:** CuPyはNumPyとほぼ同等のAPIで使える**GPU対応ライブラリ**で、NVIDIA CUDA上で動作します。CuPyの`cupy.fft`モジュールではNumPyのFFT関数と同じインターフェースでGPU上のFFT計算が可能です ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=CuPy%20covers%20the%20full%20Fast,CuPy%20provides%20additional%20features%20to))。内部ではNVIDIAの提供するcuFFT（CUDA Fast Fourier Transform library）を呼び出しており、GPUハードウェア向けに最適化された高速なFFT計算が行われます ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=CuPy%20covers%20the%20full%20Fast,CuPy%20provides%20additional%20features%20to))。つまり、コード上は`cp.fft.fft`等と書くだけで、自動的にGPU上で大規模並列計算が行われる形になります。SciPyの`scipy.fft`に相当する機能も`cupyx.scipy.fft`として一部提供されており ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=CuPy%20covers%20the%20full%20Fast,CuPy%20provides%20additional%20features%20to))、SciPyのバックエンドとしてCuPyを使うことも可能です。CuPy自体はCUDA対応GPUが必要であり、環境構築には対応するCUDA Toolkitに合わせたCuPyのインストールが必要です。精度に関しては、CuPyは基本的に**単精度(32-bit)および倍精度(64-bit)**に対応しています。さらに実験的ですが**半精度(16-bit)FFT**もサポートしており、対応ハードウェアでは単精度の約2倍速度で動作するとの報告があります ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=half,the%20necessary%20infrastructure%20for%20half))。これはNumPyにはない特徴です（NumPyは半精度FFT未対応 ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=supported%20hardware%20it%20can%20be,the%20necessary%20infrastructure%20for%20half))）。総じて、CuPyは**大規模データの処理や機械学習分野**でGPU計算をPythonから容易に利用するために設計されたライブラリであり、FFT計算もその一部として強力です。

**指定できるパラメータ:** CuPyのFFT関数（`cupy.fft.fft`, `cupy.fft.ifft`など）は、基本的に**NumPyのFFT関数と同じ引数**を受け付けます。例えば`cupy.fft.fft(x, n, axis, norm)`のようにFFT長`n`や軸`axis`、正規化モード`norm`を指定できます（現時点で`norm`引数が使えるかはバージョンによりますが、新しいCuPyではNumPy互換で実装されています）。**窓関数**も内部では扱いませんので、必要ならNumPy同様に事前に信号に掛けておきます。また、多次元FFTや実数FFT（rfft/irfft）も対応しています。加えてCuPyでは、CUDAの**ストリーム**を管理して非同期実行したり、**バッチFFT**（複数のFFTを一度に計算）を実行することも可能です。こうした高度な使い方には、cupy側で`cufft.Plan1d`等の低レベルAPIを用いる方法もありますが、一般的には高レベルな`cupy.fft`関数群で十分でしょう。

**処理時間・パフォーマンス:** **大規模なFFT**において、CuPy/cuFFTは驚異的な性能を発揮します。GPUは多数のコアと高帯域幅メモリを持つため、数百万～数千万ポイント級のFFTでもCPUを大きく上回るスループットで計算できます。ある比較では、サイズの大きい2D FFTにおいて**cuFFTはPyFFTWより約10倍、NumPyより約100倍高速**という結果も報告されています ([FFT performance using NumPy, PyFFTW, and cuFFT - John Parker](https://www.johnaparker.com/blog/fft_2d_performance#:~:text=The%20results%20on%20the%20left,of%20magnitude%20faster%20than%20NumPy))。一方で、**小さなサイズのFFT**ではGPUの起動・データ転送のオーバーヘッドが計算時間を上回り、CPUで直接計算した方が速い場合もあります ([Why is the execution time for numpy faster than cupy? - Stack Overflow](https://stackoverflow.com/questions/57060365/why-is-the-execution-time-for-numpy-faster-than-cupy#:~:text=They%20are%20essentially%20the%20same,484)) ([Why is the execution time for numpy faster than cupy? - Stack Overflow](https://stackoverflow.com/questions/57060365/why-is-the-execution-time-for-numpy-faster-than-cupy#:~:text=))。たとえば数百要素程度のFFTを1回だけ行うようなケースでは、データをGPUに送って結果を取り出す時間がもったいないため、NumPyの方が高速でしょう。従って、CuPyの真価は**大きなデータサイズ**や**多数回の繰り返し計算**に現れます。そうした状況ではGPU並列性によるスピードアップがオーバーヘッドを凌駕し、トータルの計算時間を短縮できます。また、CuPyはGPUメモリ上で他の計算とFFTを組み合わせて実行できるため、機械学習のモデル内でスペクトル変換を行う場合などでもデータ移動を最小化して高効率に処理できます。

**リアルタイム処理への適性:** CuPyをリアルタイム用途に使う場合は、**前提条件として計算対象のデータがGPU上に常駐していること**が望ましいです。例えばセンサーデータが直接GPUに取得できる環境や、GPUで前段処理を行った結果に対してFFTを掛けるケースです。そうでない場合、毎フレームCPUからGPUへのデータ転送が発生し、これが遅延のボトルネックとなりがちです。フレームごとの処理時間を極限まで抑える必要があるリアルタイムシステムでは、GPUの並列計算能力を十分に活かせない可能性があります。しかし、もし処理すべきチャンネル数が非常に多い、またはフレームサイズ自体が大きいといった状況では、GPUでまとめてFFTを実行するメリットが出てきます。CuPy/cuFFTはバッチFFT（同じ長さのFFTを多数まとめて計算）に最適化されているため、例えば数百チャンネル分のFFTを同時実行するとCPUでは到底及ばないスループットを達成できます。また、GPU内部で完結する処理であれば遅延も安定して小さいです。したがって、**低遅延かつ高スループット**が要求され、かつ**データ転送のオーバーヘッドを相殺できる規模**であれば、CuPyはリアルタイム処理にも適性があります。CUDAストリームを駆使すれば他のGPU計算との並行実行も可能で、うまく設計すればCPUでは不可能な処理性能をリアルタイムシステムで引き出せるでしょう。

**GPU対応:** CuPyそのものがGPU専用ライブラリであり、**GPU対応は完璧**です。NVIDIA製のCUDA対応GPU上で動作し、FFTに関してはCUDAの公式FFT実装であるcuFFTを利用します。AMD等のGPUについてはCuPy自体は実験的にHIPなどをサポートしつつありますが、現時点では主にNVIDIA GPUを対象としています。CuPyを使う際には、適切なCUDA環境と対応するCuPyバージョンをインストールする必要があります。

**サンプルコード:** CuPyを用いた1次元FFTの例です。GPU上の配列に対してFFTを行い、結果をCPU側に取り出して確認しています。

```python
import cupy as cp

# GPU上のデータを用意（0～7の整数）
x_gpu = cp.arange(8, dtype=cp.float32)
X_gpu = cp.fft.fft(x_gpu)       # GPU上でFFT計算
X = cp.asnumpy(X_gpu)           # 計算結果をCPU上のnumpy配列に変換

print(X)
# 出力例: [28.+0.j -4.+9.656854j -4.+4.j -4.+1.656854j -4.+0.j -4.-1.656854j ...]
```

## TensorFlow
**主な特徴:** TensorFlowはGoogleが開発した**機械学習フレームワーク**ですが、信号処理用にFFT関連のオペレーションも含んでいます。`tf.signal`モジュールにFFT機能があり、1次元から多次元までのFFTおよびIFFT、実数FFT（RFFT）などをサポートします。TensorFlowの利点は計算グラフによる最適化と自動微分機能で、FFTも計算グラフ上の演算として記述でき、必要なら**勾配（逆伝搬）**を計算することもできます。たとえば周波数領域での損失関数の勾配を求めるような応用では、フレームワークがFFT演算について自動的に微分を計算してくれます。TensorFlowはCPUとGPUの両方に対応しており、FFT演算も実行環境に応じて最適化されています。GPU上ではcuFFTを利用し、CPU上ではIntel MKLやEigen実装などで高速化されています（ビルド環境による）。TensorFlowは大規模データ処理や並列計算に強みがあり、FFTに関しても**ミニバッチ**で複数の信号を一度に処理したり、高次元の画像FFTを処理する用途に適します。ただし、ライブラリ自体が大規模なため、単純にFFTだけを目的とするにはオーバーヘッドが大きい点に注意が必要です。

**指定できるパラメータ:** TensorFlowのFFT関連関数は、`tf.signal.fft`（1次元FFT）、`tf.signal.fft2d`（2次元）、`tf.signal.fft3d`、およびそれぞれの逆変換（ifft系）、実数FFT用の`tf.signal.rfft`/`irfft`などがあります。これらの関数では**FFT長**を直接指定する引数はありませんが、実数FFTの`rfft`には代わりに`fft_length`という引数があり、これでゼロパディングや切り詰めを行うことができます ([FFT ops](https://keras.io/api/ops/fft/#:~:text=Along%20the%20axis%20RFFT%20is,dimension%20is%20padded%20with%20zeros))（指定しない場合は入力長に等しいFFTを実施 ([FFT ops](https://keras.io/api/ops/fft/#:~:text=,None))）。多次元FFTでは通常最後の軸もしくは特定の軸に対して変換を行う専用関数が用意されています（たとえば`fft2d`は最後の2軸に2D FFTを実施）。**窓関数**の適用はTensorFlow内部では行わないため、他のライブラリ同様にユーザがあらかじめ信号に窓を乗じる必要があります。**正規化**に関してTensorFlowでは、NumPyと同様の規約（逆変換で1/Nを掛ける）で固定されています。明示的に正規化を制御する引数はありません。例えば`tf.signal.fft`を行い`tf.signal.ifft`を適用すると元の配列に戻りますが、この際自動的に1/Nのスケーリングが行われています（`ifft`側で1/Nを乗じる実装）。その他、TensorFlowの演算なので`dtype`を指定することが重要です。**入力型**が`tf.complex64`や`tf.complex128`でない場合、自動的に実数として解釈され`rfft`系の結果になります。また`tf.signal.fft`に実数入力するとエラーになるため、実数しかない場合は`tf.signal.rfft`を使うか、`tf.cast(x, tf.complex64)`のように複素数型に変換してから`fft`関数を適用します。

**処理時間・パフォーマンス:** TensorFlowはGPU上で動作させた場合、cuFFTライブラリを通じて**非常に高速**にFFTを計算します ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))。特にバッチサイズが大きい（同時に処理する信号本数が多い）場合や高次元のFFTでは、その並列処理能力のおかげでスループットが高いです。一方、単発の1D FFTをCPU上で実行するような場合、前後のグラフ構築やセッション実行のオーバーヘッドがあるため、単純なNumPy実装より遅くなることもあります。実際にTensorFlow 2系ではデフォルトでEager実行（逐次実行）になりましたが、それでも内部では最適化のための準備処理があり、FFT自体の計算時間は短くても全体の関数呼び出しに要する時間は純粋なC実装を直接呼ぶより長くなりがちです。従って、小規模なFFTではTensorFlowを使う利点は少なく、逆に**大規模計算や学習モデルと組み合わせた場合**に威力を発揮します。また、TensorFlowのデバイス間メモリ管理に乗るため、GPUとCPU間のデータ転送やメモリ確保のタイミングがフレームワークに一任され、効率良く隠蔽されます。総合的には、TensorFlowのFFTは**大規模バッチ処理向き**であり、単独のFFT性能だけで見れば専門ライブラリ（FFTW等）に劣る場合もありますが、大規模並列計算や自動微分の文脈では非常に有用です。

**リアルタイム処理への適性:** 一般的にTensorFlowは訓練や大規模バッチ推論向けに設計されているため、低レイテンシのリアルタイム処理にはあまり適していません。計算グラフを使う場合は初期化に時間がかかりますし、Eager実行でも内部のスケジューリングで遅延が生じることがあります。例えば音声ストリームをフレームごとに処理するのに逐一TensorFlowを呼び出すと、フレーム毎のオーバーヘッドが大きくなる可能性があります。もっとも、TensorFlowを用いた推論システムで前処理としてFFTが必要な場合など、**リアルタイムシステム全体がTensorFlow上で構築されている**ケースでは、その一部としてFFTを使うのは合理的です。その場合でもTensorFlow Liteなどリアルタイム推論向けの軽量版を使うなどの工夫が必要でしょう。要するに、**TensorFlowはリアルタイム低遅延専用ツールではない**ため、ミリ秒単位の応答が求められる用途では採用は慎重に検討すべきです。一方で、遅延よりスループット重視（例えば1秒間に大量のFFTをまとめて処理するような状況）であれば、GPUを最大限活用できるTensorFlowは有力な選択肢となります。

**GPU対応:** TensorFlowは**GPUに完全対応**しており、FFTもGPU上で実行可能です。TensorFlowをGPU環境で動かしている場合、`tf.signal.fft`などを呼ぶと自動的に対応するGPU実装（cuFFT）が使われます ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))。デフォルトの計算精度は複素64ビット（complex64）で、これは単精度の実部・虚部で構成される複素数です ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))。複素128ビット（double precision）のFFTもCPU上では可能ですが、GPU上ではサポートが限定的で、TensorFlowの場合 GPUでcomplex128を扱えない（または非常に遅い）ため、自動的にCPUで計算されることがあります ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))。実際、TensorFlowの古いバージョンでは`tf.fft`（現`tf.signal.fft`）は入力をcomplex64に強制変換していました ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))。したがってGPU利用時は基本的に単精度でのFFTになります。いずれにせよ、TensorFlowはCPU・GPUを意識せず同じコードで実行できますが、**GPUメモリ上のデータを扱う際に最大性能を発揮**する点は押さえておきましょう。

**サンプルコード:** TensorFlowを使ったFFTの例です。実数信号に対して`tf.signal.rfft`を用いてFFTを計算し、その結果をnumpy配列として取得します。

```python
import numpy as np
import tensorflow as tf

# TensorFlow用のデータ（実数のsin波）
t = np.linspace(0, 1, 1024, endpoint=False)
x = np.sin(2 * np.pi * 60 * t)  # 60Hzの正弦波
x_tensor = tf.constant(x, dtype=tf.float32)
X_tensor = tf.signal.rfft(x_tensor)     # 実数FFTの計算（複素出力）
X = X_tensor.numpy()                   # 結果をnumpy配列に変換

print(X.shape, X.dtype)
# 出力例: (513,) complex64
```

## PyTorch
**主な特徴:** PyTorchはFacebook（現Meta）によって開発された深層学習ライブラリで、動的計算グラフとPythonライクな使いやすさが特徴です。PyTorchにもFFT機能が用意されており、`torch.fft`モジュール内でFFT/IFFTやrFFTなどが利用できます。NumPy的なインターフェースで、例えば`torch.fft.fft`が1次元FFT、`torch.fft.ifft`がその逆変換になっています。PyTorchのテンソル演算はすべて自動微分に対応しており、FFTも同様に勾配計算が可能です（もっともFFTはユニタリではないので逆変換込みで差分をとるケースが多いでしょう）。PyTorchはTensorFlowと同様にCPUとGPUに対応し、演算はC++/CUDAで最適化されています。特筆すべきは**動的な制御フロー**が可能な点で、Pythonの制御構造の中でテンソル計算を逐次行えるため、デバッグしやすくリアルタイム処理にも比較的組み込みやすいという利点があります。

**指定できるパラメータ:** PyTorchのFFT関数（`torch.fft.fft`, `torch.fft.ifft`等）は**FFT長**`n`、**変換軸**`dim`、**正規化モード**`norm`を指定できます ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=34%20Args%3A%2035%20%20,For%20the%20forward%20transform))。`norm`の指定肢や挙動はSciPy/NumPyと同様で、デフォルトが`"backward"`（逆変換で1/Nを掛ける）です ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=transform%2040%20%20%20,torch.fft.ifft%60%29%20with))。PyTorchでは入力テンソルが**実数**の場合でも`torch.fft.fft`をそのまま呼び出せ、内部で複素数に変換してからFFTが行われます ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=57%2058%20%20%20,torch.fft.fft%28t))。このとき出力は全長の複素テンソルになり、エルミート対称なスペクトルが得られます（不要な負周波数成分も含む）。もし実数入力に対して一意な半分のスペクトルだけ欲しい場合は`torch.fft.rfft`を使うことで、NumPyのrfftと同様に正の周波数成分のみの出力が得られます。**窓関数**も他のライブラリ同様に自前で掛ける必要があります。PyTorchではCPU上でもGPU上でも同じコードで動作しますが、テンソル自体をどちらのデバイスに置くか（.to('cuda')等で）を制御する形になります。

**処理時間・パフォーマンス:** PyTorchのFFTは、CPU上では高度に最適化されたライブラリ（例えばMKLやFFTWS）をバックエンドに使い、GPU上ではcuFFTを利用しています。そのため、**性能はTensorFlowと同等か場合によっては勝ります**。動的グラフゆえに余分な最適化ステップがなく、単発の演算でもオーバーヘッドが小さい傾向があります。PyTorch開発者によるドキュメントでも「CPU上では半精度（float16）および複素半精度（complex32）のFFTもサポート（ただし2のべき長に限る）」とあり ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=30%20Note%3A%2031%20%20,given%2C%20the%20input%20will%20either))、最新のGPUではFP16による高速FFTも可能です。実際、PyTorchはcomplex64とcomplex128の両方をサポートしており、必要に応じて高精度計算も行えます。もっとも、複素128でのGPU計算は非常に遅いため、GPU使用時はcomplex64が主になります。総じて、PyTorch FFTは**深層学習モデル内で使用するFFT**として最適化されており、大規模なバッチFFTやGPU演算との組み合わせで高い性能を発揮します。一方、FFT単体のベンチマークでは、PyFFTWなどに比べると若干劣ることも考えられますが、それでもNumPy標準実装よりはるかに高速です。特にGPU上の大きなFFTでは、TensorFlow同様に大きな利点があります。

**リアルタイム処理への適性:** PyTorchは動的に計算を進められるため、リアルタイム処理への組み込みもしやすいと言えます。例えば音声ストリーム処理で逐次フレームにPyTorchのテンソル変換を適用し、FFTを計算してスペクトルをPyTorch上で畳み込み、その結果を逆FFTして…といった処理を、Pythonの`for`ループ内で直接書くことができます。TensorFlowではこのような逐次処理はEagerモードでないと難しく、Eagerで行うと却ってオーバーヘッドが大きいことがありますが、PyTorchは元来Eager相当なので問題ありません。とはいえ、PyTorch自体は主にGPU資源を活用する大規模処理に焦点があるため、**極端に厳密なリアルタイム制約（例: 1ミリ秒以内の処理）**においてはオーバーヘッドがゼロではない点に注意が必要です。モデルをまたいだ遅延なども考慮すると、PyTorchだけで完結しない部分とのインターフェースで遅延が発生し得ます。しかし、一般に数十ミリ秒程度のフレーム処理であれば十分リアルタイムに動作可能であり、Python＋PyTorchだけで音声処理システムを構築している事例もあります。要するに、PyTorch FFTは**リアルタイム処理とオフライン処理の両面に適したバランス**を持っていると言えるでしょう。

**GPU対応:** PyTorchは**GPUに対応**しています。CPU上のテンソルに対して`torch.fft.fft`を呼べばCPUで計算され、GPU上のテンソル（`device='cuda'`に移したもの）に対して呼べば自動的にcuFFTを利用したGPU計算が行われます。ユーザはデータをどのデバイスに置くかだけ意識すればよく、FFT関数自体は共通です。また、PyTorchはマルチGPUもサポートしており、DistributedDataParallelなどを組み合わせれば複数GPUで並行してFFTを含む計算をさせることもできます。ただし単体のFFTを複数GPUで分割計算するようなことは通常せず、あくまでバッチをGPU間で分散させる使い方になります。

**サンプルコード:** PyTorchで1次元FFTを実行する例です。実数テンソルを用意しFFTを計算、結果のテンソルを表示します。

```python
import torch

# PyTorchテンソル（実数）
x = torch.tensor([0.0, 1.0, 0.0, -1.0])
X = torch.fft.fft(x)  # 1次元FFT

print(X)
# 出力例: tensor([0.+0.j, 1.+0.j, 0.+0.j, 1.+0.j])
```

## Numba（JIT最適化）
**主な特徴:** Numbaはライブラリというより**Pythonコードを高速化するためのJust-In-Timeコンパイラ**です。Numbaを使うと、Pythonで書いた数値計算のループ処理等を機械語にコンパイルして高速に実行できます。FFT専用の関数は提供していませんが、ユーザがFFTアルゴリズムを実装した場合にその計算を高速化したり、複数のFFT処理を並列化したりすることが可能です。たとえばPythonで単純なO(N^2)のDFTを実装すると通常は非常に遅いですが、Numbaの`@njit`デコレータを付ければC言語並みの速度で実行されます。また、NumbaはNumPy配列を扱えるため、FFTの前後処理（窓関数適用やフィルタ演算など）をまとめてJITコンパイルして効率化する、といった使い方も考えられます。依存関係はNumba本体（LLVMベース）とNumPyのみで、軽量です。

**指定できるパラメータ:** Numba自体にはFFTの機能がないため、パラメータはユーザ実装次第です。たとえば自前でコoley–TukeyのFFTを実装すれば、再帰的にサイズを処理するような柔軟性を盛り込むこともできますし、固定サイズ専用に書いてしまうこともできます。NumbaでJIT化した関数は、コンパイル時にデータ型や配列サイズが決定され最適化されます（可変長にも対応しますが、最初の呼び出し時にコンパイルが走る仕組みです）。一般に、**FFT長**や**軸**といった情報はユーザが関数内で扱うことになり、Numbaがそれを解釈するわけではありません。また、NumbaのJIT関数内から**ライブラリ関数を呼ぶ**ことも制限があります。残念ながらNumPyやSciPyのFFT関数はNumbaのサポートリストに含まれておらず、そのままでは`njit`コンパイルできません ([python - How to make discrete Fourier transform (FFT) in numba.njit? - Stack Overflow](https://stackoverflow.com/questions/62213330/how-to-make-discrete-fourier-transform-fft-in-numba-njit#:~:text=From%20the%20numba%20listings%20of,second%20case%20seems%20quite%20normal))。そのため、NumbaでFFTをやりたい場合は**自前でアルゴリズムを記述**する必要があります。一部、有志が作成したNumba対応のFFT実装（例：Rocket-FFT ([Rocket-FFT makes Numba aware of numpy.fft and scipy.fft ... - GitHub](https://github.com/styfenschaer/rocket-fft#:~:text=Rocket,8%2C%200%2C%203%2C%203))）もありますが、ここでは一般論として説明します。

**処理時間・パフォーマンス:** Numbaで最適化したFFT相当コードの性能は、実装内容とデータサイズに強く依存します。自前で最適なFFTアルゴリズム（O(N log N)）を実装できたとしても、それがFFTWやcuFFTほどチューニングされている可能性は低く、純粋な速度では劣るでしょう。しかし、Pythonで実装した単純なアルゴリズムでも、NumbaでJITコンパイルすれば**Pythonインタプリタのオーバーヘッドを排除**できるため、**数十倍以上の速度向上**が見込めます。また、Numbaは`parallel=True`オプションでデータ並列化も可能です。例えば「多数の小さなFFTを複数スレッドで同時計算する」ような並列処理をユーザが書けば、Numbaがスレッド並列にコンパイルしてくれます。ただし前述の通り、内部で`np.fft`などを呼ぶのはサポート外であり ([multithreading - Multithread many FFT operations in Python / NUMBA? - Stack Overflow](https://stackoverflow.com/questions/68755160/multithread-many-fft-operations-in-python-numba#:~:text=To%20to%20a%20convolution%20%2F,functions%20of%20SciPy%20or%20NumPy))、純粋Pythonのループ処理部分のみが並列化対象になります。つまり、高速なライブラリ呼び出し部分は並列化できない点に注意が必要です。一方で、Numbaは**GPU向けのコード**（CUDAカーネル）を書くこともできます。CUDA Pythonを用いればGPU上で実行されるカスタムFFTカーネルを記述することも理論上可能です。しかし、FFTのように複雑でメモリアクセスパターンの重要なアルゴリズムをGPUで一から実装するのは非常に難しく、性能チューニングも容易ではありません。現実的には、GPUに関してはCuPyやカスタムCUDA Cコードに任せ、NumbaはCPU上での最適化に用いるのが一般的です。総合すると、Numbaを使えば「**ライブラリと純Pythonコードの中間**」のような性能が得られ、特定のニッチな要求に合わせて最適化を試せる利点があります。

**リアルタイム処理への適性:** NumbaでJIT化したコードは、うまく書けばC/C++で書いたのと同等の速度になります。そのため、リアルタイム処理のボトルネックがPythonの遅さにある場合、Numba導入は極めて有効です。例えば毎フレームごとに特殊なスペクトル操作を行うような処理をPythonだけで記述していたケースでは、Numbaでコンパイルするだけで処理落ちしなくなる、といったことがあります。FFTそのものについて言えば、前述のようにNumbaでライブラリ並みの高速化をするのはハードルが高いですが、**「FFT + α」の処理**を一体化してコンパイルすることによりデータのやり取りや関数呼び出し回数を減らし、結果的に遅延を縮めることは可能です。例えば窓掛け・FFT・周波数ドメインフィルタ・逆FFTまでを一つのNumba関数で実装すれば、途中でPythonに戻らない分だけ低遅延になります。また、ガベージコレクションの発生も抑えられ安定した実行時間が得られるでしょう。リアルタイム処理では**一度コンパイルしてしまえば繰り返し高速に動作**するNumbaの特性は好適です。留意点として、Numba関数の初回実行時にはコンパイルが走るため僅かな遅延があります（事前にダミー呼び出ししてウォームアップ可能）。さらに、Numbaでの最適化の度合いはコードの書き方によって大きく変わります。メモリアクセスのパターンやループの使い方次第では最適化しきれない場合もあるため、プロファイリングしながら調整が必要です。以上より、Numbaは**リアルタイムDSPの自作実装を後押しする強力なツール**ですが、その効果は開発者の実装力に左右される部分もあります。

**GPU対応:** 前述のとおり、Numbaには**GPU向けコンパイル**機能（CUDA Python）があり、カスタムCUDAカーネルを書くことでGPUを利用可能です。ただし、高品質なFFTカーネルを自作するのは難易度が高く、現実的には既存のcuFFTを使う方が賢明です。Numbaが得意とするのは、GPU計算で不足する部分を補完する軽量なカーネルを書くことです。例えばデータの転置処理や簡単な並列フィルタリングなどは自作カーネルで対応し、肝心のFFT計算自体はCuPy/cuFFTに任せる、といった使い分けが考えられます。2025年現在、Numba本体にGPU向けFFT計算を簡単に行うためのユーティリティは提供されていません。したがって、**Numba単体ではFFTのGPU実行はすぐには実現できない**というのが実情です（自力でCUDAカーネルを書く場合は別ですが）。まとめると、Numbaは主にCPU上での最適化に向いており、GPUでFFTをしたい場合は他ライブラリとの併用を検討すべきです。

**サンプルコード:** Numbaを使って**直接DFTを実装**し、高速化する例を示します。長さNの複素離散信号に対し、直接計算でDFTを求める関数を`@njit`でJITコンパイルしています（アルゴリズムはO(N^2)でありFFTではありませんが、動作確認目的の簡単な例です）。

```python
import numpy as np
from numba import njit

@njit
def dft(x):
    N = x.shape[0]
    X = np.empty(N, dtype=np.complex128)
    for k in range(N):
        s = 0+0j
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            s += x[n] * np.exp(angle)
        X[k] = s
    return X

# サンプル信号（複素数列）
x = np.array([1+0j, 2+1j, 0-1j, -1+2j])
print(dft(x))
# 出力例（numpy.fft.fftと同じ結果）: [2.+2.j 3.-0.j 0.-4.j 1.-0.j]
```

上記の`dft`関数はNumbaによってコンパイルされているため、Pythonでそのまま二重ループを回すよりも格段に高速に動作します（Nが大きくなるほど差が開きます）。このようにNumbaを使えば、自作アルゴリズムであってもコンパイル最適化によってスピードアップでき、用途に合わせた柔軟な実装が可能です。

## ライブラリの機能・性能比較

各ライブラリの特徴をまとめた比較表を示します。

| ライブラリ          | 主な特徴・柔軟性 ⭐                        | パラメータ設定 🛠️                           | パフォーマンス 🔧                           | リアルタイム適性 ⏱️                                  | GPU対応 🎮                   |
|---------------------|----------------------------------------|----------------------------------------|----------------------------------------|-------------------------------------------------|--------------------------|
| **NumPy**           | Python標準のFFT実装。高精度（64bit）計算。単純で使いやすいAPI。追加依 ([Discrete Fourier Transform (numpy.fft) — NumPy v2.2 Manual](https://numpy.org/doc/stable/reference/routines.fft.html#:~:text=Type%20Promotion))58】。<br>多次元FFTや基本的な補助関数を提供。 | `n`（FFTサイズ）、`axis`（軸指定）を指 ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=n%20int%2C%20optional))84】。ゼロパディング・トリミング自 ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=Length%20of%20the%20transformed%20axis,specified%20by%20axis%20is%20used))81】。<br>窓関数パラメータは無し（必要なら事前に適用）。正規化は固定（逆変換で1/N）。 | 単一スレッド実装（デフォ ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))11】。2のべき長で高速。<br>素因数が複雑な長さでは効率低下し、FFTW等に劣る場 ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))08】。<br>使いやすさ重視で大規模最適化はなし。 | 呼び出し毎に計算、オーバーヘッド小～中。事前プランなし。<br>小～中規模FFTなら問題なく低遅延。大量の連続FFT処理ではオーバーヘッド影響大。<br>リアルタイム用途では十分な速度だが最適化余地あり。 | **非対応**（CPU計算のみ）。GPU利用にはCuPy等を併用。 |
| **SciPy**           | NumPy互換の新FFTモジュール。バグ修正や機能追加が ([Fourier Transforms With scipy.fft: Python Signal Processing – Real Python](https://realpython.com/python-scipy-fft/#:~:text=contains%20more%20features%20and%20is,should%20use%20the%20SciPy%20implementation))L4】。<br>単精度FFT対応、PocketFFTによるSIMD活 ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))11】。<br>DCT/DSTなど特殊変換も実装。 | `n`, `axis`/`axes`指定可。`norm`で正規化モード指定（“forward”/“backward”/“orth ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=norm))95】。<br>`workers`引数で並列FFTタスク数を指 ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=workers%20int%2C%20optional))09】（バッチ処理用）。<br>窓関数パラメータ無し（別途`scipy.signal`で対応）。 | PocketFFTによりNumPy版より高速化（特に多次元FFTで ([Switch from numpy.fft to scipy.fft? · Issue #6440 · dask/dask · GitHub](https://github.com/dask/dask/issues/6440#:~:text=I%20noticed%20that%20the%20,precision%20transforms))11】。<br>Bluesteinアルゴリズム採用で任意長でもO(N log N ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=FFT%20,padding%20the%20input%20using%20next_fast_len))62】。<br>複数FFTを`workers`で並列実行可。単発FFT性能はNumPy同等。 | NumPy同様都度計算だが、高速長へのパディングで高速 ([fft — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#:~:text=,padding%20the%20input%20using%20next_fast_len))63】。<br>プラン再利用機能は無し。バックエンド差替えでPyFFTWの最適化を利 ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=Since%20SciPy%20v1,set_backend%28%29%20can%20be%20used))77】。<br>リアルタイムでは安定した性能だが、追加の工夫で更に高速化余地。 | **非対応**（CPU計算のみ）。※バックエンド機構でCuPyを登録すればGPU ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=import%20cupy%20as%20cp%20import,fft))83】。 |
| **PyFFTW**          | FFTWライブラリへのPythonインターフェース。事前プランニング ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=FFTW%20is%20a%20very%20fast,fastest%20way%2C%20so%20called%20planning))25】。<br>NumPy/SciPy互換のAPIあり、既存コードに組込 ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=Quick%20and%20easy%3A%20the%20pyfftw,module%2012))47】。<br>メモリアラインメントやスレッド設定など高度な調整可。 | FFTWのプラン作成時にサイズ・型・変換方向を指定。<br>`pyfftw.interfaces`利用時はNumPy/SciPyと同じ引数（n, axis等）。<br>`pyfftw.config`でグローバルなスレッド数やプラン精度を設 ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=match%20at%20L442%20pyfftw,4))37】。<br>窓関数はライブラリ外で適用。 | 非常に高速。特に非2冪長でNumPy比数倍以 ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=The%20pyFFTW%20interfaces%20API%20provides,two%20cases))08】。<br>FFTW計画により繰返し計算で圧倒的効率（キャッシュ ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=After%20re,type%20and%20size%20are%20reused))37】。<br>NUM_THREADS指定でマルチコ ([Overview and A Short Tutorial — pyFFTW 0.15.0+2.g9bbc6da documentation](https://pyfftw.readthedocs.io/en/latest/source/tutorial.html#:~:text=,threaded%29%20pyfftw.config.NUM_THREADS%20%3D%20multiprocessing.cpu_count))21】。初回計画に時間かかるが再利用で高速。 | リアルタイム用途に最適。プランを事前作成することで毎回の呼び出しオーバーヘッ ([Faster Fast Fourier Transforms in Python - QMUL ITS Research Blog](https://blog.hpc.qmul.ac.uk/pyfftw/#:~:text=After%20re,type%20and%20size%20are%20reused))37】。<br>連続フレーム処理でも安定した低遅延。初回のプラン計算のみ注意（起動時に実行）。 | **非対応**（CPU計算のみ）。GPU利用不可（FFT自体はCPUベース）。 |
| **CuPy**            | NumPy互換のGPUライブラリ。`cupy.fft`でcuFFT ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=CuPy%20covers%20the%20full%20Fast,CuPy%20provides%20additional%20features%20to))60】。<br>大規模並列計算が可能で、NumPy/SciPyのコードを置換するだけでGPU加速。<br>半精度FFTもサポート（実 ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=half,the%20necessary%20infrastructure%20for%20half))15】。 | NumPy FFTと同じAPI（fft, ifft, fftn, rfft等）で`n`, `axis`, `norm`指定可（NumPy互換） 。<br>CUDAストリーム管理やカスタムPlanも可能（上級者向け）。<br>窓関数は自前適用。 | 大規模データで最高速。GPUの圧倒的並列性でCPU ([FFT performance using NumPy, PyFFTW, and cuFFT - John Parker](https://www.johnaparker.com/blog/fft_2d_performance#:~:text=The%20results%20on%20the%20left,of%20magnitude%20faster%20than%20NumPy))18】。<br>小規模ではPCIe転送等のオーバーヘッドでCPUより遅い場 ([Why is the execution time for numpy faster than cupy? - Stack Overflow](https://stackoverflow.com/questions/57060365/why-is-the-execution-time-for-numpy-faster-than-cupy#:~:text=They%20are%20essentially%20the%20same,484))64】。<br>適切な条件下でNumPy比10～100倍の速度 ([FFT performance using NumPy, PyFFTW, and cuFFT - John Parker](https://www.johnaparker.com/blog/fft_2d_performance#:~:text=The%20results%20on%20the%20left,of%20magnitude%20faster%20than%20NumPy))18】。 | GPU上で処理が完結する場合は低遅延（演算自体は高速）。<br>ただし毎フレームCPU-GPU間転送があると遅延増大。<br>多数チャネルや大きなフレームの並列処理に威力発揮。 | **対応**（要CUDA対応GPU）。NVIDIA GPU上でcuFFTによ ([Fast Fourier Transform with CuPy — CuPy 13.4.1 documentation](https://docs.cupy.dev/en/stable/user_guide/fft.html#:~:text=CuPy%20covers%20the%20full%20Fast,CuPy%20provides%20additional%20features%20to))60】。<br>AMD GPUは未サポート（将来的にはhipFFT対応の可能性）。 |
| **TensorFlow**      | 機械学習フレームワーク。`tf.signal`でFFT提供（グラフ計算&自動微分対応）。<br>大規模バッチ・高次元データ向けに最適化。深層学習と統合可能。 | `tf.signal.fft`/`ifft`, `fft2d`, `rfft`等で次元別に関数用意。<br>`rfft`には`fft_length`指定でゼロパディング ([FFT ops](https://keras.io/api/ops/fft/#:~:text=Along%20the%20axis%20RFFT%20is,dimension%20is%20padded%20with%20zeros))01】。軸指定は関数固有。<br>正規化設定なし（内部で逆変換時1/N）。窓関数無し。 | GPU上ではcuFFT使用 ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))83】。CPU上も最適化ライブラリ利用。<br>しかし単発FFT時はフレームワークのオーバーヘッド大きめ。<br>大規模並列処理で真価を発揮。 | リアルタイム用途にはオーバーヘッドが大きく不向き。<br>ただしモデル内処理として組込む場合は許容可能。<br>逐次処理よりバッチ処理向き。軽量化にはTensorFlow Lite等必要。 | **対応**（GPU使用時は自動でcuF ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))83】。<br>デフォルト複素64bit精度（GPUはcomplex64のみ ([python - The result of fft in tensorflow is different from numpy - Stack Overflow](https://stackoverflow.com/questions/47214508/the-result-of-fft-in-tensorflow-is-different-from-numpy#:~:text=You%27re%20right%2C%20the%20difference%20is,in%20tensorflow%20and%20numpy))83】。 |
| **PyTorch**         | 動的計算グラフの深層学習フレームワーク。`torch.fft`でFFT提供。<br>NumPyライクな操作性で自動微分対応。モデル内でFFT活用可能。 | `n`, `dim`（軸）, `norm`を指 ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=34%20Args%3A%2035%20%20,For%20the%20forward%20transform))69】（デフォルト“backwar ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=transform%2040%20%20%20,torch.fft.ifft%60%29%20with))77】。<br>実数入力にそのままfft可（複素出力に ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=57%2058%20%20%20,torch.fft.fft%28t))99】。`rfft`等もあり。<br>窓関数パラメータ無し（事前適用）。 | CPUではMKL/FFTW等で最適化、GPUではcuFFT使用。<br>半精度FFTもサポート（GPUで2のべき長 ([PyTorch: torch/fft/__init__.py | Fossies](https://fossies.org/linux/pytorch/torch/fft/__init__.py#:~:text=30%20Note%3A%2031%20%20,given%2C%20the%20input%20will%20either))63】。<br>TensorFlow同様大規模計算で高速。小規模でも動的実行でオーバーヘッド小さめ。 | 動的フローでリアルタイム処理に組込みやすい。<br>Pythonループ内で逐次FFTも可能（オーバーヘッド小）。<br>フレーム毎処理も比較的低遅延だが、単体FFT目的での利用は過剰な場合も。 | **対応**（GPU上のテンソルに対して自動GPU計算）。<br>複数GPUもサポート（データ並列）。半精度GPU計算対応。 |
| **Numba**           | PythonコードをJITコンパイルして高速化。FFT専用ではないが、自作FFTや周辺処理を高速化可能。<br>柔軟な実装をそのまま最適化できる。 | ライブラリ関数としてのFFTは未サポート（np.fftはnjit ([python - How to make discrete Fourier transform (FFT) in numba.njit? - Stack Overflow](https://stackoverflow.com/questions/62213330/how-to-make-discrete-fourier-transform-fft-in-numba-njit#:~:text=From%20the%20numba%20listings%20of,second%20case%20seems%20quite%20normal))L4】。<br>自前実装に応じ自由にパラメータ処理可能。並列化オプション（parallel=True）で複数FFT並列 ([multithreading - Multithread many FFT operations in Python / NUMBA? - Stack Overflow](https://stackoverflow.com/questions/68755160/multithread-many-fft-operations-in-python-numba#:~:text=To%20to%20a%20convolution%20%2F,functions%20of%20SciPy%20or%20NumPy))12】。<br>GPU用にcuda.jitでカーネルを書くことも可能。 | Pythonコードのオーバーヘッドを除去し、C並み速度に。<br>単純DFTでも数十倍速くなる例多数。専用ライブラリには純計算速度で劣る場合が多い。<br>並列化やベクトル化次第で高性能が得られるがチューニングは手動。 | リアルタイム処理向けに自作アルゴリズムを最適化可能。<br>FFT前後の処理を含め一体化すれば遅延削減に寄与。<br>初回実行時のコンパイル遅延に注意。最適化度合いはコード依存。 | **間接対応**（GPU向けコードを記述可能だが、FFT用ビルトインなし）。<br>CUDA PythonでGPUカーネルを書く必要あり。既存cuFFTと併用が現実的。 |

※⭐主な特徴・柔軟性欄の引用では、NumPyのfloat32入力がfloat64に昇格す ([Discrete Fourier Transform (numpy.fft) — NumPy v2.2 Manual](https://numpy.org/doc/stable/reference/routines.fft.html#:~:text=Type%20Promotion))58】などを示しています。