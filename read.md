PythonでのWAVファイル読み込みライブラリの比較
主要なPythonライブラリによるWAV音声ファイルの読み込み方法について、SciPy.io.wavfile, SoundFile (PySoundFile), librosa, wave (標準ライブラリ), audioread, torchaudio, pydub のそれぞれを比較します。以下では各ライブラリごとに、読み込みの使いやすさ、正規化の挙動、出力データ形式、対応フォーマット、依存関係、他ライブラリとの統合性、および読み込み性能についてまとめ、最後に比較表で整理します。
SciPy.io.wavfile
読み込み方法: SciPyのscipy.io.wavfileモジュールは関数呼び出しでシンプルに使えます。wavfile.read('ファイルパス')とするだけで、サンプルレートとデータ配列をタプルで返します。使い方は非常に簡単ですが、読み込み機能は基本的なPCM WAVEに限定されています。部分的な読み込み（ストリーミング）には対応しておらず、一度に全データを読み込みます（ただしmmap=Trueオプションでメモリマップ読み込みは可能です）。
正規化: 自動正規化は行われません。WAVファイルの量子化ビット深度に応じた整数値データをそのまま返します​
DOCS.SCIPY.ORG
。例えば16ビットPCMなら範囲[-32768, 32767]のint16値、24ビットPCMならint32型（24bit音声を上位24ビットに保持）で値を返します​
DOCS.SCIPY.ORG
。入力が32-bit float PCMの場合はfloat32で読み込まれ、その値が±1を超えていてもクリップされません​
DOCS.SCIPY.ORG
。[-1,1]へのスケーリングは行われないため、必要ならユーザ側で整数値を32768.0などで割る処理が必要です。
出力形式: NumPy配列（numpy.ndarray）で返されます。モノラル音声の場合は1次元配列、ステレオなど複数チャンネルの場合は形状が(Nsamples, Nchannels)の2次元配列になります。データ型(dtype)は入力ファイルに依存し、8bitならuint8、16bitならint16、24/32bitならint32、32bit浮動小数ならfloat32になります​
DOCS.SCIPY.ORG
。データ型に応じた生の値を保持しており、float型以外はそのままPCM整数表現です。
対応フォーマット: 非圧縮PCMのWAVのみサポートします​
DOCS.PYTHON.ORG
。チャンネル数はモノラルからステレオ以上でも読み込めます（配列の第二軸にチャンネルが展開されます）。サンプリング周波数も制限なく、ファイルに記録された値をそのまま取得します。対応ビット深度は1～64bitの任意の整数PCMに対応しており、24bitも内部で32bitに展開して読み取れます​
DOCS.SCIPY.ORG
。圧縮されたWAV（ADPCMやμ-lawなど）や非PCMのWAVはサポートされません​
DOCS.SCIPY.ORG
。
依存関係: SciPyライブラリに含まれる機能であり、NumPyに依存しますが、追加のマルチメディア用Cライブラリは不要です。SciPy自体はライブラリサイズが大きめですが、科学技術計算環境では一般的にインストールされています。実装はPythonベースで、NumPyを使ってPCMバイト列を配列に変換しています。
他ライブラリとの統合: 出力がNumPy配列であるため、そのままNumPy/SciPyの信号処理や機械学習ライブラリ（TensorFlow/PyTorch等へ変換）で利用可能です。特別な統合機能はありませんが、シンプルな構造ゆえに他の処理系へデータを渡しやすいです。
読み込み速度・パフォーマンス: 手軽さ重視の実装であり、C言語による最適化は特に行われていません。中程度のサイズのファイルであれば十分実用的な速度ですが、大きなファイルではPythonでのデータ処理分だけ若干遅くなる可能性があります。I/O自体はディスク読み込み速度が支配的で、大きな差は出にくいものの、巨大ファイルを多数処理する用途では後述のSoundFile等よりやや低速になる場合があります。
読み込みサンプルコード:
python
コピーする
編集する
from scipy.io import wavfile

# WAVファイルを読み込む
sr, data = wavfile.read('input.wav')

print(sr)        # サンプリング周波数（整数）
print(data.shape, data.dtype)  # NumPy配列の形状とデータ型
# 例: (441000, 2), dtype=int16 など
SoundFile (PySoundFile)
読み込み方法: soundfileモジュール（PySoundFile）ではsf.read('ファイルパス')により、音声配列とサンプルレートを取得できます​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。使い方はSciPyとほぼ同じく簡潔です。加えて、開始位置やフレーム数を指定して部分読み込みする引数（start, frames）や、読み込みデータ型を指定するdtypeオプションも利用できます​
PYTHON-SOUNDFILE.READTHEDOCS.IO
​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。大量データを扱う場合には、ブロック単位で読み込むblocks()や、メモリに載せず逐次処理する方法も用意されています。
正規化: デフォルトで浮動小数（倍精度）に変換して読み込みます。そのため、整数PCMのWAVファイルであっても、自動的に**[-1.0, 1.0]範囲にスケーリング**された浮動小数値となります​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。例えば16bit PCMの場合、sf.readの戻り値はfloat64型（既定）で振幅が-1.0～+1.0程度の値になります（最大振幅は1.0近辺）​
STACKOVERFLOW.COM
。この挙動によりユーザが手動で正規化する手間が省けます。なお、dtypeを'int16'や'int32'に指定すれば整数値そのままで読み込めますが、その場合も自動スケーリングは行われません​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。
出力形式: NumPy配列として出力されます。デフォルトではfloat64のnumpy.ndarrayですが、dtype='float32'などとすれば単精度にすることも可能です​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。モノラルの場合は1次元配列、ステレオ以上の場合は形状(フレーム数, チャンネル数)の2次元配列（オプションでalways_2d=Trueを指定すればモノラルでも2次元にできます）​
PYTHON-SOUNDFILE.READTHEDOCS.IO
​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。整数で読み込む場合はint16やint32の配列になります。例: 16bitステレオPCMのWAVを読み込むと、デフォルトではfloat64型の形状(N,2)配列（各要素は-1～1の値）として得られます。
対応フォーマット: libsndfileライブラリに対応する幅広いフォーマットの読み書きが可能です​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。WAV（PCM 8～32bit, 浮動小数点）やFLAC, OGG, AIFFなど多くのフォーマットに対応しています。チャンネル数もモノラルから多チャンネルまで制限なく扱えます。サンプリング周波数もファイルに記録された値を正確に取得します。24bit WAVのような特殊なPCMも問題なく読み込めます（実際内部ではint32に展開して保持）​
STACKOVERFLOW.COM
。非線形PCM（μ-law/A-law等）や一部の圧縮音声はlibsndfileが対応していれば読み込めますが、一般的なPCM WAVE/FLACであれば網羅的にカバーしています。
依存関係: C言語製のlibsndfileライブラリをPythonから呼び出すラッパーであり、インストール時にlibsndfileが必要です​
PYTHON-SOUNDFILE.READTHEDOCS.IO
（pip経由で自動インストールされる環境もあります）。内部実装はCFFIを用いているため、高速ながらもPythonから手軽に使えます​
PYTHON-SOUNDFILE.READTHEDOCS.IO
。依存ライブラリは比較的軽量で、SciPyよりはインストールサイズが小さいです。
他ライブラリとの統合: librosaが内部で音声読み込みに利用するなど、他の音声処理ライブラリとの親和性も高いです​
LIBROSA.ORG
。出力がNumPy配列なので、SciPyやNumPyによる信号処理はもちろん、PyTorchなどへテンソル変換して機械学習に用いることも簡単です。torchaudioでもバックエンドとしてsoundfileを利用可能で、お好みのフレームワークにデータを受け渡ししやすいです。単体でも入出力・簡易編集ができる汎用ライブラリとして位置付けられています。
読み込み速度・パフォーマンス: パフォーマンスは良好です。音声データのデコード/読み出しはほぼlibsndfile（C実装）が担うため、大きなファイルでも効率よく読み込めます。ディスクI/Oがボトルネックにならない範囲では、Python処理のオーバーヘッドも小さく抑えられています。ただし超大量のファイルを高速に読みまくるケースでは、後述のtorchaudio（C++実装の最適化あり）などと比べてわずかに遅いという報告もあります​
GITHUB.COM
。一般的にはほとんどの用途で十分高速であり、部分的なデータ読み出しやメモリマップによる効率的処理も可能です。
読み込みサンプルコード:
python
コピーする
編集する
import soundfile as sf

# WAVファイルを読み込み（デフォルトではfloat64で取得）
data, sr = sf.read('input.wav')
print(sr)               # サンプリング周波数
print(data.shape, data.dtype, data.min(), data.max())
# 例: (441000, 2), float64, -0.5, 0.4 など（振幅は-1〜1程度）
librosa
読み込み方法: librosaライブラリは音楽情報処理向けに高機能ですが、librosa.load('ファイルパス', sr=対象サンプルレート, mono=ブール)で手軽に音声読み込みできます​
LIBROSA.ORG
。戻り値は波形データ配列とサンプリング周波数です。初期値ではsr=22050が指定されており、自動的にこのレートへリサンプリングされます​
LIBROSA.ORG
。元のレートで読み込みたい場合はsr=Noneと明示します​
LIBROSA.ORG
。またデフォルトmono=Trueのため、ステレオ音声は自動的にモノラル合成（各チャンネルの平均）されます​
LIBROSA.ORG
。mono=Falseを指定すればマルチチャンネルそのままで読み込めます。こうしたデフォルト動作に注意すれば、非常に簡潔に使えるAPIです。
正規化: librosaで読み込むと常に浮動小数点型のnumpy配列になり、振幅は-1.0～+1.0に正規化されています。内部的には、可能であればsoundfileを利用し、それが不可ならaudioreadで16bit PCMデータを取得してから 1/32768スケーリング（約±1範囲に収まるよう調整）します​
LIBROSA.ORG
​
STACKOVERFLOW.COM
。したがって、出力配列はオーディオ信号として扱いやすい実数表現になっています。クリッピングや追加の正規化処理は行われませんが、リサンプリング時にエイリアシング防止のフィルタが適用されます。
出力形式: NumPy配列 (dtype=np.float32がデフォルト) で返されます​
STACKOVERFLOW.COM
。デフォルトではモノラル化されるため1次元配列になりますが、mono=Falseの場合は形状が(チャンネル数, サンプル数)の2次元配列になります​
LIBROSA.ORG
。例えばステレオのWAVをmono=Falseで読み込むとshape=(2, N)の配列となり、チャンネルごとの波形が別配列として得られます​
LIBROSA.ORG
。常にfloat32型なので、後段の処理との互換性も高く（深層学習フレームワークはfloat32を標準とします）、数値計算精度も適度です。
対応フォーマット: 対応フォーマットは非常に広範囲です。librosa自体は音声ファイル解析のラッパーであり、実際の読み込みはsoundfileまたはaudioreadに委ねています​
LIBROSA.ORG
。soundfile経由ならWAV/FLAC/OGGなど幅広く、audioread経由ならFFmpegで扱えるほぼ全ての音声形式（MP3等）を読み込めます​
LIBROSA.ORG
。WAVに限定すれば、PCM 8–32bitや浮動小数点WAVも問題なく処理できます。チャンネル数やサンプリングレートも制限なく、入力ファイルの情報を取得できます。librosaは音楽信号処理用途であり、MP3/OGGなど圧縮音声からでもnumpy配列にデコード可能です。
依存関係: librosaは機能が豊富な分、依存ライブラリも多いです。NumPy/SciPyはもちろん、音響特徴量計算にsklearnやnumbaを利用する場合があります。音声読み込みには前述のsoundfileかaudioreadが必要です（librosaインストール時に一緒に入ることが多い）。サイズは比較的大きく、インポート時に多少時間がかかります。ただしlibrosa一つで読み込みから信号処理（STFTやメルスペクトログラム計算など）まで幅広く賄える利点があります。
他ライブラリとの統合: librosaは自前で多くの音声処理機能を提供するため、他ライブラリに依存せず完結できます。出力はNumPy配列なので、必要であれば他のライブラリへ渡すことも容易です。例えば得られた波形をPyTorchテンソルに変換してtorchaudioの機能を使う、といったことも可能です。また、近年librosaは内部のaudioread依存を減らし、PySoundFileベースに移行しつつあります​
LIBROSA.ORG
。これは深層学習フレームワークなどとの併用時にも、精度（16bit以上の精度保持）や速度で有利になるためです。
読み込み速度・パフォーマンス: librosa自身はPythonで実装されていますが、読み込みは内部で高性能なライブラリを使うため速度面でも概ね良好です。soundfileを使う場合は前述の通りC実装の高速さが得られます。audioread+FFmpeg経由の場合もFFmpeg自体は高速ですが、librosaが小さなバッファを順次読み込んで結合する処理を行うため、若干のオーバーヘッドがあります。それでも通常の音楽長（数分程度）であれば問題ない速度です。ただ、librosaはデフォルトでリサンプリング処理を行う点に注意が必要です​
LIBROSA.ORG
。高品質（soxrライブラリ）のリサンプルは計算コストが大きいため、長時間音源を大量に処理する場合はsr=Noneにする、あるいは他の方法でリサンプルする方が効率的です。総じて、単なる読み込み用途であればlibrosaはオーバーヘッドがあり、後述の専門ライブラリに劣りますが、音響解析まで含めたワークフローでは利便性がパフォーマンス面のデメリットを上回ります。
読み込みサンプルコード:
python
コピーする
編集する
import librosa

# WAVファイルを読み込み（元のサンプルレート維持、ステレオ保持）
y, sr = librosa.load('input.wav', sr=None, mono=False)
print(sr)           # サンプリング周波数（元の値）
print(y.shape, y.dtype)  
# 例: (2, 441000), float32  （2チャンネル、データ型float32）
wave（標準ライブラリ）
読み込み方法: Python標準ライブラリの**waveモジュールでもWAVファイルを扱えます​
DOCS.PYTHON.ORG
。使い方は低レベルで、wave.open('ファイルパス', 'rb')でファイルを開き、readframes(N)メソッドでバイト列データを取得します​
REALPYTHON.COM
。取得したバイト列をPython組み込みのstructモジュールやnumpy.frombufferで数値配列に変換して利用します。例えば16bitステレオPCMの場合、frames = wav.readframes(wav.getnframes())で全フレームの生データを読み込み、np.frombuffer(frames, dtype='<i2')（リトルエンディアンint16）でNumPy配列化し、さらに.reshape(-1, 2)で2チャンネルに整形します。標準ライブラリだけで完結しますが、このように手動の処理**が必要で、他のライブラリに比べると手軽さは劣ります。
正規化: 一切行われません。生のPCMデータをそのままバイト列で取得するだけなので、ユーザが適切に数値変換・スケーリングする必要があります。例えば上記のようにint16に変換すれば-32768～32767の整数値となり、[-1,1]に正規化したければそれらを32768で割る処理を自前で実装します。waveモジュール自体はサンプル値を解釈せずバイト列として提供するだけです。
出力形式: waveモジュールのreadframesはbytesオブジェクトを返します。そのため直接NumPy配列は得られませんが、上記のようにnumpyでバッファを解釈するか、struct.unpackなどでPythonのリストに変換できます。最終的なデータ型や形状はユーザの変換次第ですが、多くの場合はint16やint32のNumPy配列にします。マルチチャンネルの場合、PCMデータはインターリーブ形式（例: L,R,L,R,...）で並んでいるため、適宜reshape等でチャンネル軸を分離します。標準モジュールにはチャンネル分離や配列化の補助はない点に注意が必要です。
対応フォーマット: 非圧縮PCMのWAVのみサポートします​
DOCS.PYTHON.ORG
。Pythonドキュメントにも「PCMエンコードされたWAVファイルしか扱えない」と明記されています​
DOCS.PYTHON.ORG
。すなわち、8, 16, 24, 32bitのリニアPCMに対応します（Python 3.12以降では拡張ヘッダWAVE_FORMAT_EXTENSIBLEにも対応）​
DOCS.PYTHON.ORG
。浮動小数点WAV（32bit float）については公式には触れられていませんが、PCMではないため対応外です。チャンネル数やサンプリングレートの情報はgetnchannels(), getframerate()メソッドで取得できます。圧縮形式（例: ADPCMやMP3埋め込みのWAVなど）は扱えません​
STACKOVERFLOW.COM
。
依存関係: Python標準ライブラリのみで動作し、追加の依存はありません。非常に軽量ですが、機能も限定的です。C拡張も使っていないため、環境を選ばず実行できます。プラットフォームに関係なく動く反面、高度な最適化はされていません。
他ライブラリとの統合: 出力が汎用のPython型（bytesや組み込みarray）なので、そのままでは他ライブラリでの利用は不便です。他のライブラリと組み合わせる場合、一旦NumPy配列などに変換する必要があります。ただ、標準モジュールである安心感から、簡易なスクリプトで外部ライブラリを導入せず音声を扱いたい場合に使われる程度で、他の音声処理ライブラリと直接連携するケースは多くありません。
読み込み速度・パフォーマンス: 必要最低限の処理しか行わないため、読み込み自体のオーバーヘッドは小さいです。バイト列取得後にNumPyでメモリコピーする処理もそれほど重くありません。したがって単発のファイルを読み込む分には速度上大きな問題はありません。しかし、多数のファイルを扱う場合や大容量データのストリーミング処理では、高水準ライブラリの最適化（C実装のバッファ処理や並列処理など）が効かない分、徐々に見劣りしてきます。また、waveモジュールはファイルのメタデータ取得やチャンク操作に限定的なため、例えば長時間ファイルの一部だけを効率的に読む、といった用途には向きません。総じて、シンプルさと互換性優先であり、パフォーマンスや機能は限定的です。
読み込みサンプルコード:
python
コピーする
編集する
import wave
import numpy as np

with wave.open('input.wav', 'rb') as wf:
    nframes = wf.getnframes()
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()  # 2 bytes for 16-bit PCM
    frames = wf.readframes(nframes)

# 16-bit PCMの場合の例（リトルエンディアン）
data = np.frombuffer(frames, dtype='<i2')  
if nchannels > 1:
    data = data.reshape(-1, nchannels)
print(data.shape, data.dtype)
# 例: (441000, 2), dtype=int16
audioread
読み込み方法: audioreadは様々なバックエンドを用いて音声データをデコードする裏方的なライブラリです​
PYPI.ORG
​
PYPI.ORG
。直接使う場合、audioread.audio_open('ファイルパス')でファイルオブジェクトを取得し、イテレータとして音声データのバッファを順次読み出します​
PYPI.ORG
。例えば:with audioread.audio_open('input.wav') as f: print(f.channels, f.samplerate); for buf in f: ...のように使い、ループ内で得られるbuf（bytes型データ）を蓄積していきます​
PYPI.ORG
。一括でnumpy配列を得る便利関数は用意されていないため、ユーザ側でバッファを結合して配列化する処理が必要です。手順が多く、librosaなどで自動利用されることが多いです（librosaでは内部でこの処理を行っています）。
正規化: 自動的な振幅正規化は行われません。audioreadが返すデータバッファはリトルエンディアン16-bit PCMのバイト列に統一されています​
PYPI.ORG
。つまり、どのようなフォーマットの入力でも、一旦16bit整数のPCMデータにデコードして提供する設計です​
PYPI.ORG
。このため、24bitや32bit浮動小数点の入力では精度が劣化（16bit相当に丸め）します​
LIBROSA.ORG
。ユーザがnumpy配列に変換した後、必要なら /32768.0 などで[-1,1]範囲に正規化します。ライブラリ側は正規化もクリッピングもしない生PCM提供に徹しています。
出力形式: ライブラリ自体はbytesデータの塊をイテレーションで返すのみです。各バッファは既定で約4096フレーム程度のPCMバイト列になります。ユーザがそれらをつなげてNumPyのint16型などに変換することで初めて数値配列となります。モノラル・ステレオ等のチャンネル情報はf.channels属性で取得でき、PCMバッファ中には各チャンネルのサンプルが交互に格納されています（典型的なPCMのインターリーブ形式）。したがって、最終的には例えばnp.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, channels)のような処理で形状を整えることになります。
対応フォーマット: 非常に広範です。audioreadは複数のバックエンド（GStreamer、Core Audio、MAD、FFmpeg、Python標準のwave等）を試行し、読み込み可能な方法でデコードします​
PYPI.ORG
​
PYPI.ORG
。そのため、WAVに限らずMP3やAACなどFFmpegが対応するあらゆる音声フォーマットを扱えます​
PYPI.ORG
。実質的に入力フォーマットの制約はありません（バックエンドがインストールされていれば）。WAVについて言えば、PCMはもちろん、圧縮形式のWAVもffmpeg経由でデコードできます。ただし出力は前述の通り16bit PCMに統一される点に注意が必要です​
LIBROSA.ORG
。
依存関係: 純粋なPythonパッケージですが、実際の動作には外部プログラム/ライブラリが必要です。デフォルトではシステムのFFmpegコマンドまたはGStreamerを呼び出してデコードを行います​
PYPI.ORG
。MacであればCore Audio、LinuxであればGStreamerが使えればそれを利用し、見つからなければFFmpegを使用、それも無ければPythonのwaveモジュールにフォールバックする仕組みです​
PYPI.ORG
。したがって環境によっては追加のインストールが必要です。librosaではpipインストール時に一緒に入るため意識しないことも多いですが、単体で使う際はバックエンドの用意に留意が必要です。
他ライブラリとの統合: librosaが標準で利用していたことから知名度があります。librosa 0.10以降は非推奨になりましたが、それ以前はMP3読み込みなどで重要な役割を果たしていました​
LIBROSA.ORG
。直接他のライブラリがaudioreadを使うケースは少ないですが、「どんな音声ファイルでも読み込める」汎用デコーダとして、用途に応じて組み込まれることがあります。出力がNumPy対応ではないため、実質的には裏方として動き、最終出力は他ライブラリ側でnumpy配列化されることが多いです。
読み込み速度・パフォーマンス: デコード処理自体は多くの場合FFmpegなどの高速な外部ツールが行うため、音声変換そのものは効率的です。しかし、audioreadは小さなチャンクのPythonループでデータを渡すため、大量データ処理ではオーバーヘッドが蓄積しやすいです。librosaの開発者も、「audio読み込みの性能上の制限は主にPython側にある」と言及しています​
GITHUB.COM
。また16bitPCMへの変換固定という仕様上、他のライブラリよりデータ量が減るとはいえ精度損失があります。総じて、汎用性重視のため性能面では専用ライブラリに及ばないケースもあります。librosaでは将来的にaudioreadサポートを廃止予定であり​
LIBROSA.ORG
、近年では必要な場合のみ使われる位置付けです。
読み込みサンプルコード:
python
コピーする
編集する
import audioread
import numpy as np

# audioreadでWAVを読み込み、NumPy配列にする例
with audioread.audio_open('input.wav') as f:
    sr = f.samplerate
    ch = f.channels
    # 全バッファを結合してから一括変換
    raw_bytes = b''.join([buf for buf in f])
audio = np.frombuffer(raw_bytes, dtype=np.int16)
if ch > 1:
    audio = audio.reshape(-1, ch)
print(sr, audio.shape, audio.dtype)
# 例: 44100, (441000, 2), int16
torchaudio
読み込み方法: torchaudioはPyTorchエコシステムの音声処理ライブラリで、torchaudio.load('ファイルパス')によりPyTorch Tensorとサンプルレートを返します​
PYTORCH.ORG
。使い方はlibrosa同様にシンプルで、デフォルトでは全フレームを読み込みます（frame_offsetやnum_frames引数で部分読み込み可能）​
PYTORCH.ORG
。PyTorchのTensorとして得られるため、そのままGPUに載せたりミニバッチ化したりといった操作がスムーズに行えます。デフォルト引数channels_first=Trueにより、戻りTensorのshapeは[チャンネル, 時間フレーム]となります​
PYTORCH.ORG
。
正規化: torchaudioのloadはデフォルトで正規化を行いますが、その内容は「整数PCMを浮動小数に変換する」ことです​
PYTORCH.ORG
​
PYTORCH.ORG
。normalize=True（デフォルト）の場合、整数型のWAVはtorch.float32テンソルに変換され、値が[-1.0,1.0]程度の範囲にスケーリングされます​
PYTORCH.ORG
。これはボリューム正規化ではなく型変換である点に注意が必要です​
PYTORCH.ORG
。一方、引数normalize=Falseを指定すると、入力が整数PCMの場合は元のビット深度の整数Tensorが得られます​
PYTORCH.ORG
。例えば16bit PCMならdtype=torch.int16、24bit PCMならint24がないためint32、32bit PCMならint32のTensorとなります​
PYTORCH.ORG
。この場合、値は-32768～32767等の整数範囲を保持しています。浮動小数点WAVやFLAC/MP3などの場合、normalizeフラグに関係なくfloat32 Tensorで読み込まれます​
PYTORCH.ORG
。つまり、torchaudioは必要に応じて柔軟にデータ型と正規化有無を選べる設計です。
出力形式: PyTorchのTensorで出力されます。デフォルトではfloat32型で、shapeは(チャンネル, フレーム数)です​
PYTORCH.ORG
。チャンネル数1の場合も2次元（1, N）のTensorになります（PyTorchでは1次元Tensorだとチャンネル軸情報を失うため）。channels_first=Falseを指定すれば(shape: 時間×チャンネル)のTensorになります​
PYTORCH.ORG
。NumPy配列が必要な場合は.numpy()メソッドで変換可能ですが、torchaudioはTensorで統一することでその後のニューラルネットワーク処理などへの連携をスムーズにしています。また、後述の通りSoXやSoundFileなど複数のバックエンドを選択可能ですが、出力の形式は統一されています。
対応フォーマット: torchaudioは内部で複数のI/Oバックエンドを持ち、デフォルトではSoX (torchaudioソースコードに組み込まれたSoX-IO) を使用します。SoXはWAV (PCM/浮動小数点)はもちろん、MP3やFLACなど主要なフォーマットに対応しています。さらにFFmpegをバックエンドに指定することもでき、これを用いればFFmpeg対応フォーマット全般を読み込めます​
PYTORCH.ORG
。また、PySoundFileをバックエンドに選ぶことも可能です​
PYTORCH.ORG
。標準のSoXバックエンドでも24bit PCMや32bit float PCMに対応しており、前述のように24bitはint32 Tensorにマップされます​
PYTORCH.ORG
。チャンネル数やサンプリングレートもファイルから取得し、そのままsrとして返します。多チャンネル音声も問題なくTensor化できます。要約すると、torchaudioは非常に幅広い音声フォーマットに対応しており、特にWAVに関して不足はありません。
依存関係: PyTorch本体と、その音声拡張ライブラリであるtorchaudioを必要とします。PyTorch自体が大規模なため、単に音声読み込みだけの目的で導入するのは重量級です。しかし既にPyTorchを使用しているプロジェクトでは追加負荷なく利用できます。torchaudio内部では、C++で実装されたSoXライブラリやFFmpegラッパーを利用しており、これらはtorchaudioインストール時に同梱されています。そのため、ユーザが別途SoXやFFmpegをインストールする必要は基本的にありません（ただしFFmpegバックエンド利用時は環境によってはシステムのffmpegを参照する場合もあります）。要するに、機械学習用途向けにPyTorch + torchaudio環境が前提となります。
他ライブラリとの統合: torchaudioはPyTorchとの統合が最大の特徴です。読み込んだTensorはそのままGPU上のモデルに入力したり、PyTorchのDataset/DataLoaderで複数の音声をミニバッチ化したりできます。また、torchaudio自体にSTFTやフィルタ、データ増強（時間伸縮・ピッチシフト等）などの機能も揃っており、librosaに頼らずPyTorch内で完結した音響前処理が可能です。さらに、torchaudioはSoundFileをバックエンドにできるように設計されており、SoundFile経由で読み込んだNumPy配列をtorch.Tensorに変換する処理を内部的に行うこともできます。そのため、他のライブラリとのデータ受け渡しも比較的容易です（例：SoundFileで読みNumPy配列化 -> torch.from_numpy()でTensor化）。ただし基本的にはPyTorchユーザ以外にtorchaudioを使うメリットは少なく、独立した音声処理ライブラリというよりはPyTorchユーザ向けの専用ツールと言えます。
読み込み速度・パフォーマンス: torchaudioは高いパフォーマンスを発揮します。C++実装のSoX or FFmpegを直接呼び出してデコードし、その結果を即座にTensor領域に格納するため、中間のPython処理が極力排されています。特に大量のファイルをGPU上で処理する深層学習ワークロードでは、データローディングがボトルネックになりがちですが、torchaudio + DataLoaderの組み合わせはマルチスレッドで効率よくバッチを組み立てることが可能です。ベンチマークによれば、SoundFile+NumPyで読み込んでからTensor化するよりもtorchaudioで直接Tensorとして読み込む方が若干高速であった例もあります​
GITHUB.COM
。これは内部でのメモリ転送や変換コストを削減しているためです。また、部分読み込み（frame_offset/num_frames）もサポートしており、大きなファイルから必要な区間だけ取得して訓練する、といった効率的なI/Oもできます。総じて、大量データ処理に耐えうる設計と速度を持っている点がtorchaudioの強みです。
読み込みサンプルコード:
python
コピーする
編集する
import torchaudio

# WAVファイルを読み込み（torch.Tensorで取得）
waveform, sr = torchaudio.load('input.wav')
print(sr)                # サンプリング周波数
print(waveform.shape, waveform.dtype)  
# 例: torch.Size([2, 441000]), torch.float32
# 必要に応じて NumPy 配列に変換:
np_wave = waveform.numpy()
pydub
読み込み方法: pydubは音声ファイルのカット・結合やエフェクト処理に便利な高レベルライブラリです。読み込みにはAudioSegment.from_wav('ファイルパス')などのクラスメソッドを使用します。WAVに限らず、from_fileで拡張子に応じたファイルを読み込めます。戻り値はAudioSegmentオブジェクトで、波形データやメタ情報を保持しています。直接NumPy配列ではありませんが、audio_segment.get_array_of_samples()メソッドで内部のPCMデータにアクセスできます​
GITHUB.COM
。このメソッドはPythonのarray型（標準配列モジュール）を返し、各サンプル値のシーケンスを取得できます​
GITHUB.COM
。例えばステレオ音声なら、「左1サンプル, 右1サンプル, 左2サンプル, 右2サンプル,...」の順で並んだarray('h')（int16型配列）が得られます​
GITHUB.COM
。それをnp.array()でNumPy配列に変換すれば数値演算に使えます。
正規化: 自動正規化は行われません。AudioSegment内部ではソースのPCM値をそのまま保持しています。get_array_of_samples()が返す配列の値は、例えば16bit音源なら-32768～32767の整数値です。振幅を0～±1の実数に正規化するには、自身でnumpy.array(..., dtype=np.float32) / 32768.0といった処理を行う必要があります​
GITHUB.COM
。pydubは主にオーディオ編集用途を想定しており、ラウドネス正規化などは別途メソッド（apply_gain等）で提供していますが、波形そのものの数値正規化は組み込まれていません。従って、分析のためにデータを取り出す際は必要に応じて手動でスケーリングします。
出力形式: get_array_of_samples()の戻り値はPythonのarray型（例えばarray('h')）です​
GITHUB.COM
。これをNumPyに変換すると1次元のint型配列になります。モノラル音声ならそのまま時系列配列ですが、ステレオなどの場合は左右チャネルが交互に並んだ1次元配列となるため、NumPy配列にした後で.reshape(-1, 2)のように2次元に整形する必要があります。この操作によりshape=(Nsamples, 2)の配列（列ごとに左/右チャンネル）が得られます。pydub自体はチャンネルを分離するメソッドも提供しており、例えばaudio_segment.split_to_mono()で各チャンネルのAudioSegmentを取り出すことも可能です。その場合それぞれに対してget_array_of_samples()を呼べばチャンネル別のデータを取得できます。データ型は通常int16相当ですが、ファイルのbit深度によります。もっとも、後述の通り内部的に多くの形式は16bitに変換されるため、実質常に16bit整数として扱われます。
対応フォーマット: pydubはFFmpegもしくはlibavを利用してほぼ全ての音声フォーマットを読み込みます。AudioSegment.from_fileは拡張子や引数formatでファイル種別を判断し、適切にデコードを試みます。WAVに関してはPCMはもちろん、非PCMや圧縮音声(WAVラップのMP3など)もffmpeg経由で展開できます。チャンネル数やサンプリングレートもファイルから引き継がれ、audio.frame_rate, audio.channelsプロパティで確認できます。ただし内部表現は特定のフォーマットに統一されます。デフォルトではPCMの場合はソースと同じbit深度を維持しますが、FFmpegのデコード出力によっては16bitに変換されることがあります。特に24bit WAVは直接扱えず16bitに落とされることが知られています​
STACKOVERFLOW.COM
。pydub自体は24bit対応していないため、別途24bit→16bit変換が入ります。そのため高音質データの精度保持には不向きです。一般的な16bitステレオ44.1kHz WAVやMP3程度であれば問題なく扱えます。
依存関係: ffmpegもしくはlibavのインストールが必要です。pydubはそれらのコマンドラインツールを呼び出してデコードを行います。Pythonライブラリ単体では動作せず、環境にffmpegが無い場合はエラーになります。pydub自体は純Pythonで軽量ですが、背後にプロ音響用の強力なツールを必要とします。また、一部機能ではPython標準のaudioopモジュールを使用しています。これはC実装のため高速ですが、主にモノラル/ステレオの範囲でしか動作確認されていません。つまり、5.1chなど多チャンネル音声の処理は限定的です。総じて、pydub = Pythonからffmpegを操るラッパーと考えると分かりやすいでしょう。
他ライブラリとの統合: pydubはオーディオデータの加工や変換に特化しており、例えば波形を切り出してファイルに保存したり、MP3に一括変換したりといった用途に便利です。他の分析ライブラリと直接統合する想定はあまりなく、NumPy配列が必要な場合は上述の手順で取り出す必要があります。pydubを用いてデータ増強（重ね合わせやフェード処理）を行い、その結果をNumPyにして機械学習に入力、といったパイプラインを構築することも可能ですが、リアルタイム性や大規模データセット処理には向きません。基本的には単発の音声編集操作やファイル形式変換などに優れたツールです。他ライブラリとの連携はファイルの入出力経由で行うのが一般的です。
読み込み速度・パフォーマンス: ffmpegをバックエンドに使うためデコードそのものは高速ですが、一度全体をAudioSegment（メモリ上）に展開する仕様のため、大きなファイルではメモリ消費が大きくなります。読み込み速度は、ffmpegプロセスの起動時間が加算されるため、小さなファイルを大量に処理する場合にはオーバーヘッドが無視できなくなります。逆に長時間のファイルであっても一括で読み込むため、その後の処理はメモリ内で高速に行えます（代償として最初に時間を要する）。メタデータを取得するだけでもデコードが走るため、ライブラリによってはpydubでの情報取得は他より遅いと報告されています​
GITHUB.COM
。総合すると、対話的・単発的な処理には便利だが、効率面では専門ライブラリに劣る部分があります。しかしpydubの強みは性能よりも「簡潔に高度な編集ができる」点にあり、用途によっては十分実用的な速度です。
読み込みサンプルコード:
python
コピーする
編集する
from pydub import AudioSegment
import numpy as np

# WAVファイルをAudioSegmentとして読み込み
audio = AudioSegment.from_wav('input.wav')
samples = np.array(audio.get_array_of_samples())  # Python array -> NumPy配列
# ステレオの場合、2チャンネルに整形
if audio.channels > 1:
    samples = samples.reshape(-1, audio.channels)
print(audio.frame_rate, samples.shape, samples.dtype)
# 例: 44100, (441000, 2), int16
ライブラリ比較まとめ
以上の各ライブラリの特徴を、読み込み方法・APIの容易さ, 正規化の挙動, 出力データ形式, 対応フォーマット（チャンネル数・サンプルレート・ビット深度）, 依存関係, 統合性, パフォーマンスの観点でまとめた比較表を示します。

了解しました。PythonでWAVファイルを読み込む主要なライブラリについて、それぞれの特徴（読み込み方式、正規化の有無、出力形式、依存関係、対応フォーマットなど）を詳しく比較し、サンプルコードも含めてまとめます。特に信号の正規化の挙動に注目して整理します。

調査が完了次第、比較表とともにお伝えします。

# PythonでのWAVファイル読み込みライブラリの比較

主要なPythonライブラリによるWAV音声ファイルの読み込み方法について、**SciPy.io.wavfile**, **SoundFile (PySoundFile)**, **librosa**, **wave** (標準ライブラリ), **audioread**, **torchaudio**, **pydub** のそれぞれを比較します。以下では各ライブラリごとに、読み込みの使いやすさ、正規化の挙動、出力データ形式、対応フォーマット、依存関係、他ライブラリとの統合性、および読み込み性能についてまとめ、最後に比較表で整理します。

## SciPy.io.wavfile

- **読み込み方法**: SciPyの`scipy.io.wavfile`モジュールは関数呼び出しでシンプルに使えます。`wavfile.read('ファイルパス')`とするだけで、**サンプルレート**と**データ配列**をタプルで返します。使い方は非常に簡単ですが、読み込み機能は基本的なPCM WAVEに限定されています。部分的な読み込み（ストリーミング）には対応しておらず、一度に全データを読み込みます（ただし`mmap=True`オプションでメモリマップ読み込みは可能です）。

- **正規化**: 自動正規化は行われません。WAVファイルの量子化ビット深度に応じた整数値データをそのまま返します ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=WAV%20files%20can%20specify%20arbitrary,bit%20and%20higher%20is%20signed))。例えば16ビットPCMなら範囲[-32768, 32767]の`int16`値、24ビットPCMなら`int32`型（24bit音声を上位24ビットに保持）で値を返します ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=WAV%20files%20can%20specify%20arbitrary,bit%20and%20higher%20is%20signed))。入力が32-bit float PCMの場合は`float32`で読み込まれ、その値が±1を超えていてもクリップされません ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=IEEE%20float%20PCM%20in%2032,are%20not%20clipped))。[-1,1]へのスケーリングは行われないため、必要ならユーザ側で整数値を32768.0などで割る処理が必要です。

- **出力形式**: **NumPy配列**（numpy.ndarray）で返されます。モノラル音声の場合は1次元配列、ステレオなど複数チャンネルの場合は形状が(Nsamples, Nchannels)の2次元配列になります。データ型(dtype)は入力ファイルに依存し、8bitなら`uint8`、16bitなら`int16`、24/32bitなら`int32`、32bit浮動小数なら`float32`になります ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=WAV%20files%20can%20specify%20arbitrary,bit%20and%20higher%20is%20signed))。データ型に応じた生の値を保持しており、float型以外はそのままPCM整数表現です。

- **対応フォーマット**: **非圧縮PCMのWAV**のみサポートします ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。チャンネル数はモノラルからステレオ以上でも読み込めます（配列の第二軸にチャンネルが展開されます）。サンプリング周波数も制限なく、ファイルに記録された値をそのまま取得します。対応ビット深度は1～64bitの任意の整数PCMに対応しており、24bitも内部で32bitに展開して読み取れます ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=WAV%20files%20can%20specify%20arbitrary,bit%20and%20higher%20is%20signed))。**圧縮**されたWAV（ADPCMやμ-lawなど）や非PCMのWAVはサポートされません ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=IEEE%20float%20PCM%20in%2032,are%20not%20clipped))。

- **依存関係**: SciPyライブラリに含まれる機能であり、**NumPy**に依存しますが、追加のマルチメディア用Cライブラリは不要です。SciPy自体はライブラリサイズが大きめですが、科学技術計算環境では一般的にインストールされています。実装はPythonベースで、NumPyを使ってPCMバイト列を配列に変換しています。

- **他ライブラリとの統合**: 出力がNumPy配列であるため、そのまま**NumPy/SciPyの信号処理**や**機械学習ライブラリ**（TensorFlow/PyTorch等へ変換）で利用可能です。特別な統合機能はありませんが、シンプルな構造ゆえに他の処理系へデータを渡しやすいです。

- **読み込み速度・パフォーマンス**: **手軽さ重視の実装**であり、C言語による最適化は特に行われていません。中程度のサイズのファイルであれば十分実用的な速度ですが、大きなファイルではPythonでのデータ処理分だけ若干遅くなる可能性があります。I/O自体はディスク読み込み速度が支配的で、大きな差は出にくいものの、巨大ファイルを多数処理する用途では後述のSoundFile等よりやや低速になる場合があります。

**読み込みサンプルコード**: 

```python
from scipy.io import wavfile

# WAVファイルを読み込む
sr, data = wavfile.read('input.wav')

print(sr)        # サンプリング周波数（整数）
print(data.shape, data.dtype)  # NumPy配列の形状とデータ型
# 例: (441000, 2), dtype=int16 など
```

## SoundFile (PySoundFile)

- **読み込み方法**: `soundfile`モジュール（PySoundFile）では`sf.read('ファイルパス')`により、**音声配列とサンプルレート**を取得できます ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,0.11516333%5D%5D%29%20%3E%3E%3E%20samplerate%2044100))。使い方はSciPyとほぼ同じく簡潔です。加えて、開始位置やフレーム数を指定して部分読み込みする引数（`start`, `frames`）や、読み込みデータ型を指定する`dtype`オプションも利用できます ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=soundfile.read%28file%2C%20frames%3D,source%5D%20%2018)) ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,optional%29%20%E2%80%93))。大量データを扱う場合には、ブロック単位で読み込む`blocks()`や、メモリに載せず逐次処理する方法も用意されています。

- **正規化**: **デフォルトで浮動小数（倍精度）に変換して読み込み**ます。そのため、整数PCMのWAVファイルであっても、自動的に**[-1.0, 1.0]範囲にスケーリング**された浮動小数値となります ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,optional%29%20%E2%80%93))。例えば16bit PCMの場合、`sf.read`の戻り値は`float64`型（既定）で振幅が-1.0～+1.0程度の値になります（最大振幅は1.0近辺） ([audio - Python's SoundFile: soundfile.write and clipping - Stack Overflow](https://stackoverflow.com/questions/69388531/pythons-soundfile-soundfile-write-and-clipping#:~:text=The%20array%20,1%29%2C%20and%20I%20use))。この挙動によりユーザが手動で正規化する手間が省けます。なお、`dtype`を`'int16'`や`'int32'`に指定すれば整数値そのままで読み込めますが、その場合も自動スケーリングは行われません ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,optional%29%20%E2%80%93))。

- **出力形式**: **NumPy配列**として出力されます。デフォルトでは`float64`のnumpy.ndarrayですが、`dtype='float32'`などとすれば単精度にすることも可能です ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,optional%29%20%E2%80%93))。モノラルの場合は1次元配列、ステレオ以上の場合は形状(フレーム数, チャンネル数)の2次元配列（オプションで`always_2d=True`を指定すればモノラルでも2次元にできます） ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=Returns%3A)) ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,dimensional%20array%20anyway))。整数で読み込む場合は`int16`や`int32`の配列になります。**例**: 16bitステレオPCMのWAVを読み込むと、デフォルトではfloat64型の形状(N,2)配列（各要素は-1～1の値）として得られます。

- **対応フォーマット**: **libsndfile**ライブラリに対応する幅広いフォーマットの読み書きが可能です ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=file%20using%20soundfile.read%28%29.%20The%20,below%20about%20writing%20OGG%20files))。WAV（PCM 8～32bit, 浮動小数点）やFLAC, OGG, AIFFなど多くのフォーマットに対応しています。チャンネル数もモノラルから多チャンネルまで制限なく扱えます。サンプリング周波数もファイルに記録された値を正確に取得します。**24bit WAV**のような特殊なPCMも問題なく読み込めます（実際内部ではint32に展開して保持） ([python - Process 24bit wavs using pydub - Stack Overflow](https://stackoverflow.com/questions/48847933/process-24bit-wavs-using-pydub#:~:text=It%20has%20been%20some%20months,using%20that%20data))。非線形PCM（μ-law/A-law等）や一部の圧縮音声はlibsndfileが対応していれば読み込めますが、一般的なPCM WAVE/FLACであれば網羅的にカバーしています。

- **依存関係**: C言語製の**libsndfile**ライブラリをPythonから呼び出すラッパーであり、インストール時にlibsndfileが必要です ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=The%20,audio%20data%20as%20NumPy%20arrays))（pip経由で自動インストールされる環境もあります）。内部実装は**CFFI**を用いているため、高速ながらもPythonから手軽に使えます ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=The%20soundfile%20module%20is%20an,soundfile.readthedocs.io))。依存ライブラリは比較的軽量で、SciPyよりはインストールサイズが小さいです。

- **他ライブラリとの統合**: librosaが内部で音声読み込みに利用するなど、他の音声処理ライブラリとの親和性も高いです ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=If%20the%20codec%20is%20supported,SoundFile%20object))。出力がNumPy配列なので、SciPyやNumPyによる信号処理はもちろん、PyTorchなどへテンソル変換して機械学習に用いることも簡単です。torchaudioでもバックエンドとしてsoundfileを利用可能で、お好みのフレームワークにデータを受け渡ししやすいです。単体でも入出力・簡易編集ができる汎用ライブラリとして位置付けられています。

- **読み込み速度・パフォーマンス**: **パフォーマンスは良好**です。音声データのデコード/読み出しはほぼlibsndfile（C実装）が担うため、大きなファイルでも効率よく読み込めます。ディスクI/Oがボトルネックにならない範囲では、Python処理のオーバーヘッドも小さく抑えられています。ただし超大量のファイルを高速に読みまくるケースでは、後述のtorchaudio（C++実装の最適化あり）などと比べてわずかに遅いという報告もあります ([Soundfile read is slower than torchaudio.load with soundfile backend · Issue #376 · bastibe/python-soundfile · GitHub](https://github.com/bastibe/python-soundfile/issues/376#:~:text=I%20will%20say%20that%20soundfile,improving%20performance%2C%20I%27m%20all%20ears))。一般的にはほとんどの用途で十分高速であり、部分的なデータ読み出しやメモリマップによる効率的処理も可能です。

**読み込みサンプルコード**:

```python
import soundfile as sf

# WAVファイルを読み込み（デフォルトではfloat64で取得）
data, sr = sf.read('input.wav')
print(sr)               # サンプリング周波数
print(data.shape, data.dtype, data.min(), data.max())
# 例: (441000, 2), float64, -0.5, 0.4 など（振幅は-1〜1程度）
```

## librosa

- **読み込み方法**: **librosa**ライブラリは音楽情報処理向けに高機能ですが、`librosa.load('ファイルパス', sr=対象サンプルレート, mono=ブール)`で手軽に音声読み込みできます ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=librosa.load%28path%2C%20,class%20%27numpy.float32%27%3E%2C%20res_type%3D%27soxr_hq%27%29%5Bsource%5D%20%206))。戻り値は**波形データ配列**と**サンプリング周波数**です。初期値では`sr=22050`が指定されており、自動的にこのレートへリサンプリングされます ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Audio%20will%20be%20automatically%20resampled,sr%3D22050))。元のレートで読み込みたい場合は`sr=None`と明示します ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Audio%20will%20be%20automatically%20resampled,sr%3D22050))。またデフォルト`mono=True`のため、ステレオ音声は自動的に**モノラル合成**（各チャンネルの平均）されます ([Multi-channel — librosa 0.10.2 documentation](http://librosa.org/doc/0.10.2/multichannel.html#:~:text=%28monaural%29%20signals%20are%20processed,in%20the%20signal%2C%20which%20is))。`mono=False`を指定すればマルチチャンネルそのままで読み込めます。こうしたデフォルト動作に注意すれば、非常に簡潔に使えるAPIです。

- **正規化**: librosaで読み込むと**常に浮動小数点型のnumpy配列**になり、**振幅は-1.0～+1.0に正規化**されています。内部的には、可能であればsoundfileを利用し、それが不可ならaudioreadで16bit PCMデータを取得してから **1/32768スケーリング**（約±1範囲に収まるよう調整）します ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=%3E%3E%3E%20,05%5D%2C%20dtype%3Dfloat32%29%20%3E%3E%3E%20sr%2022050)) ([audio - Python's SoundFile: soundfile.write and clipping - Stack Overflow](https://stackoverflow.com/questions/69388531/pythons-soundfile-soundfile-write-and-clipping#:~:text=x%20%3D%20np.array%28,x.wav))。したがって、出力配列はオーディオ信号として扱いやすい実数表現になっています。クリッピングや追加の正規化処理は行われませんが、リサンプリング時にエイリアシング防止のフィルタが適用されます。

- **出力形式**: **NumPy配列** (`dtype=np.float32`がデフォルト) で返されます ([python - Why are scipy and librosa different for reading wav file? - Stack Overflow](https://stackoverflow.com/questions/51489784/why-are-scipy-and-librosa-different-for-reading-wav-file/56476151#:~:text=It%27s%20a%20type%20mismatch,float32))。デフォルトではモノラル化されるため1次元配列になりますが、`mono=False`の場合は形状が(**チャンネル数**, **サンプル数**)の2次元配列になります ([Multi-channel — librosa 0.10.2 documentation](http://librosa.org/doc/0.10.2/multichannel.html#:~:text=%23%20Load%20as%20multi,load%28filename%2C%20mono%3DFalse))。例えばステレオのWAVを`mono=False`で読み込むとshape=(2, N)の配列となり、チャンネルごとの波形が別配列として得られます ([Multi-channel — librosa 0.10.2 documentation](http://librosa.org/doc/0.10.2/multichannel.html#:~:text=%23%20Load%20as%20multi,load%28filename%2C%20mono%3DFalse))。常にfloat32型なので、後段の処理との互換性も高く（深層学習フレームワークはfloat32を標準とします）、数値計算精度も適度です。

- **対応フォーマット**: **対応フォーマットは非常に広範囲**です。librosa自体は音声ファイル解析のラッパーであり、実際の読み込みはsoundfileまたはaudioreadに委ねています ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Any%20codec%20supported%20by%20soundfile,will%20work))。soundfile経由ならWAV/FLAC/OGGなど幅広く、audioread経由ならFFmpegで扱えるほぼ全ての音声形式（MP3等）を読み込めます ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Any%20codec%20supported%20by%20soundfile,will%20work))。WAVに限定すれば、PCM 8–32bitや浮動小数点WAVも問題なく処理できます。チャンネル数やサンプリングレートも制限なく、入力ファイルの情報を取得できます。librosaは**音楽信号処理**用途であり、MP3/OGGなど圧縮音声からでもnumpy配列にデコード可能です。

- **依存関係**: librosaは機能が豊富な分、**依存ライブラリも多い**です。NumPy/SciPyはもちろん、音響特徴量計算に`sklearn`や`numba`を利用する場合があります。音声読み込みには前述のsoundfileかaudioreadが必要です（librosaインストール時に一緒に入ることが多い）。サイズは比較的大きく、インポート時に多少時間がかかります。ただしlibrosa一つで読み込みから信号処理（STFTやメルスペクトログラム計算など）まで幅広く賄える利点があります。

- **他ライブラリとの統合**: librosaは**自前で多くの音声処理機能**を提供するため、他ライブラリに依存せず完結できます。出力はNumPy配列なので、必要であれば他のライブラリへ渡すことも容易です。例えば得られた波形をPyTorchテンソルに変換してtorchaudioの機能を使う、といったことも可能です。また、近年librosaは内部のaudioread依存を減らし、PySoundFileベースに移行しつつあります ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Warning))。これは深層学習フレームワークなどとの併用時にも、精度（16bit以上の精度保持）や速度で有利になるためです。

- **読み込み速度・パフォーマンス**: librosa自身はPythonで実装されていますが、読み込みは内部で高性能なライブラリを使うため**速度面でも概ね良好**です。soundfileを使う場合は前述の通りC実装の高速さが得られます。audioread+FFmpeg経由の場合もFFmpeg自体は高速ですが、librosaが小さなバッファを順次読み込んで結合する処理を行うため、若干のオーバーヘッドがあります。それでも通常の音楽長（数分程度）であれば問題ない速度です。ただ、librosaは**デフォルトでリサンプリング処理**を行う点に注意が必要です ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Audio%20will%20be%20automatically%20resampled,sr%3D22050))。高品質（soxrライブラリ）のリサンプルは計算コストが大きいため、長時間音源を大量に処理する場合はsr=Noneにする、あるいは他の方法でリサンプルする方が効率的です。総じて、単なる読み込み用途であればlibrosaはオーバーヘッドがあり、後述の専門ライブラリに劣りますが、音響解析まで含めたワークフローでは利便性がパフォーマンス面のデメリットを上回ります。

**読み込みサンプルコード**:

```python
import librosa

# WAVファイルを読み込み（元のサンプルレート維持、ステレオ保持）
y, sr = librosa.load('input.wav', sr=None, mono=False)
print(sr)           # サンプリング周波数（元の値）
print(y.shape, y.dtype)  
# 例: (2, 441000), float32  （2チャンネル、データ型float32）
```

## wave（標準ライブラリ）

- **読み込み方法**: Python標準ライブラリの**`wave`モジュール**でもWAVファイルを扱えます ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。使い方は低レベルで、`wave.open('ファイルパス', 'rb')`でファイルを開き、`readframes(N)`メソッドでバイト列データを取得します ([Reading and Writing WAV Files in Python – Real Python](https://realpython.com/python-wav-files/#:~:text=match%20at%20L391%20,nframes%29))。取得したバイト列をPython組み込みの`struct`モジュールや`numpy.frombuffer`で数値配列に変換して利用します。例えば16bitステレオPCMの場合、`frames = wav.readframes(wav.getnframes())`で全フレームの生データを読み込み、`np.frombuffer(frames, dtype='<i2')`（リトルエンディアンint16）でNumPy配列化し、さらに`.reshape(-1, 2)`で2チャンネルに整形します。標準ライブラリだけで完結しますが、このように**手動の処理**が必要で、他のライブラリに比べると手軽さは劣ります。

- **正規化**: 一切行われません。**生のPCMデータ**をそのままバイト列で取得するだけなので、ユーザが適切に数値変換・スケーリングする必要があります。例えば上記のようにint16に変換すれば-32768～32767の整数値となり、[-1,1]に正規化したければそれらを32768で割る処理を自前で実装します。waveモジュール自体は**サンプル値を解釈せずバイト列として提供する**だけです。

- **出力形式**: waveモジュールの`readframes`は**bytesオブジェクト**を返します。そのため直接NumPy配列は得られませんが、上記のように**numpy**でバッファを解釈するか、**struct.unpack**などでPythonのリストに変換できます。最終的なデータ型や形状はユーザの変換次第ですが、多くの場合はint16やint32のNumPy配列にします。マルチチャンネルの場合、PCMデータは**インターリーブ形式**（例: L,R,L,R,...）で並んでいるため、適宜reshape等でチャンネル軸を分離します。標準モジュールにはチャンネル分離や配列化の補助はない点に注意が必要です。

- **対応フォーマット**: **非圧縮PCMのWAVのみ**サポートします ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。Pythonドキュメントにも「PCMエンコードされたWAVファイルしか扱えない」と明記されています ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。すなわち、8, 16, 24, 32bitのリニアPCMに対応します（Python 3.12以降では拡張ヘッダWAVE_FORMAT_EXTENSIBLEにも対応） ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。**浮動小数点WAV**（32bit float）については公式には触れられていませんが、PCMではないため対応外です。チャンネル数やサンプリングレートの情報は`getnchannels()`, `getframerate()`メソッドで取得できます。**圧縮形式**（例: ADPCMやMP3埋め込みのWAVなど）は扱えません ([How do I write a file in headerless PCM format? - Stack Overflow](https://stackoverflow.com/questions/16530485/how-do-i-write-a-file-in-headerless-pcm-format#:~:text=How%20do%20I%20write%20a,file%20in%20the%20same))。

- **依存関係**: Python標準ライブラリのみで動作し、追加の依存はありません。非常に軽量ですが、機能も限定的です。C拡張も使っていないため、環境を選ばず実行できます。プラットフォームに関係なく動く反面、高度な最適化はされていません。

- **他ライブラリとの統合**: 出力が汎用のPython型（bytesや組み込みarray）なので、そのままでは他ライブラリでの利用は不便です。他のライブラリと組み合わせる場合、一旦NumPy配列などに変換する必要があります。ただ、標準モジュールである安心感から、簡易なスクリプトで**外部ライブラリを導入せず音声を扱いたい場合**に使われる程度で、他の音声処理ライブラリと直接連携するケースは多くありません。

- **読み込み速度・パフォーマンス**: **必要最低限の処理**しか行わないため、読み込み自体のオーバーヘッドは小さいです。バイト列取得後にNumPyでメモリコピーする処理もそれほど重くありません。したがって単発のファイルを読み込む分には速度上大きな問題はありません。しかし、多数のファイルを扱う場合や大容量データのストリーミング処理では、高水準ライブラリの最適化（C実装のバッファ処理や並列処理など）が効かない分、**徐々に見劣り**してきます。また、waveモジュールはファイルのメタデータ取得やチャンク操作に限定的なため、例えば長時間ファイルの一部だけを効率的に読む、といった用途には向きません。総じて、**シンプルさと互換性優先**であり、パフォーマンスや機能は限定的です。

**読み込みサンプルコード**:

```python
import wave
import numpy as np

with wave.open('input.wav', 'rb') as wf:
    nframes = wf.getnframes()
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()  # 2 bytes for 16-bit PCM
    frames = wf.readframes(nframes)

# 16-bit PCMの場合の例（リトルエンディアン）
data = np.frombuffer(frames, dtype='<i2')  
if nchannels > 1:
    data = data.reshape(-1, nchannels)
print(data.shape, data.dtype)
# 例: (441000, 2), dtype=int16
```

## audioread

- **読み込み方法**: `audioread`は様々なバックエンドを用いて音声データをデコードする裏方的なライブラリです ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Decode%20audio%20files%20using%20whichever,The%20library%20currently%20supports)) ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,line%20interface))。直接使う場合、`audioread.audio_open('ファイルパス')`でファイルオブジェクトを取得し、イテレータとして音声データのバッファを順次読み出します ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Use%20the%20library%20like%20so%3A))。例えば:`with audioread.audio_open('input.wav') as f: print(f.channels, f.samplerate); for buf in f: ...`のように使い、ループ内で得られる`buf`（bytes型データ）を蓄積していきます ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Use%20the%20library%20like%20so%3A))。一括でnumpy配列を得る便利関数は用意されていないため、ユーザ側でバッファを結合して配列化する処理が必要です。手順が多く、librosaなどで自動利用されることが多いです（librosaでは内部でこの処理を行っています）。

- **正規化**: 自動的な振幅正規化は行われません。`audioread`が返すデータバッファは**リトルエンディアン16-bit PCMのバイト列**に統一されています ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Buffers%20in%20the%20file%20can,to%20most%20of%20the%20backends))。つまり、どのようなフォーマットの入力でも、一旦16bit整数のPCMデータにデコードして提供する設計です ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Buffers%20in%20the%20file%20can,to%20most%20of%20the%20backends))。このため、24bitや32bit浮動小数点の入力では精度が劣化（16bit相当に丸め）します ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Note))。ユーザがnumpy配列に変換した後、必要なら `/32768.0` などで[-1,1]範囲に正規化します。ライブラリ側は正規化もクリッピングもしない**生PCM提供**に徹しています。

- **出力形式**: ライブラリ自体は**bytesデータの塊**をイテレーションで返すのみです。各バッファは既定で約4096フレーム程度のPCMバイト列になります。ユーザがそれらをつなげてNumPyの`int16`型などに変換することで初めて数値配列となります。モノラル・ステレオ等のチャンネル情報は`f.channels`属性で取得でき、PCMバッファ中には**各チャンネルのサンプルが交互に格納**されています（典型的なPCMのインターリーブ形式）。したがって、最終的には例えば`np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, channels)`のような処理で形状を整えることになります。

- **対応フォーマット**: 非常に広範です。**audioreadは複数のバックエンド**（GStreamer、Core Audio、MAD、FFmpeg、Python標準のwave等）を試行し、読み込み可能な方法でデコードします ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=)) ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,for%20uncompressed%20audio%20formats))。そのため、WAVに限らずMP3やAACなど**FFmpegが対応するあらゆる音声フォーマット**を扱えます ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,bindings))。実質的に入力フォーマットの制約はありません（バックエンドがインストールされていれば）。WAVについて言えば、PCMはもちろん、圧縮形式のWAVもffmpeg経由でデコードできます。ただし出力は前述の通り16bit PCMに統一される点に注意が必要です ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Note))。

- **依存関係**: 純粋なPythonパッケージですが、**実際の動作には外部プログラム/ライブラリ**が必要です。デフォルトではシステムの**FFmpeg**コマンドまたは**GStreamer**を呼び出してデコードを行います ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,bindings))。MacであればCore Audio、LinuxであればGStreamerが使えればそれを利用し、見つからなければFFmpegを使用、それも無ければPythonのwaveモジュールにフォールバックする仕組みです ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,for%20uncompressed%20audio%20formats))。したがって環境によっては追加のインストールが必要です。librosaではpipインストール時に一緒に入るため意識しないことも多いですが、単体で使う際はバックエンドの用意に留意が必要です。

- **他ライブラリとの統合**: **librosaが標準で利用**していたことから知名度があります。librosa 0.10以降は非推奨になりましたが、それ以前はMP3読み込みなどで重要な役割を果たしていました ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Warning))。直接他のライブラリがaudioreadを使うケースは少ないですが、「どんな音声ファイルでも読み込める」汎用デコーダとして、用途に応じて組み込まれることがあります。出力がNumPy対応ではないため、実質的には**裏方**として動き、最終出力は他ライブラリ側でnumpy配列化されることが多いです。

- **読み込み速度・パフォーマンス**: デコード処理自体は多くの場合**FFmpegなどの高速な外部ツール**が行うため、音声変換そのものは効率的です。しかし、audioreadは**小さなチャンクのPythonループ**でデータを渡すため、大量データ処理ではオーバーヘッドが蓄積しやすいです。librosaの開発者も、「audio読み込みの性能上の制限は主にPython側にある」と言及しています ([Soundfile read is slower than torchaudio.load with soundfile backend · Issue #376 · bastibe/python-soundfile · GitHub](https://github.com/bastibe/python-soundfile/issues/376#:~:text=is%20different%3F))。また16bitPCMへの変換固定という仕様上、他のライブラリよりデータ量が減るとはいえ**精度損失**があります。総じて、**汎用性重視**のため性能面では専用ライブラリに及ばないケースもあります。librosaでは将来的にaudioreadサポートを廃止予定であり ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Warning))、近年では必要な場合のみ使われる位置付けです。

**読み込みサンプルコード**:

```python
import audioread
import numpy as np

# audioreadでWAVを読み込み、NumPy配列にする例
with audioread.audio_open('input.wav') as f:
    sr = f.samplerate
    ch = f.channels
    # 全バッファを結合してから一括変換
    raw_bytes = b''.join([buf for buf in f])
audio = np.frombuffer(raw_bytes, dtype=np.int16)
if ch > 1:
    audio = audio.reshape(-1, ch)
print(sr, audio.shape, audio.dtype)
# 例: 44100, (441000, 2), int16
```

## torchaudio

- **読み込み方法**: **torchaudio**はPyTorchエコシステムの音声処理ライブラリで、`torchaudio.load('ファイルパス')`により**PyTorch Tensor**と**サンプルレート**を返します ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source))。使い方はlibrosa同様にシンプルで、デフォルトでは全フレームを読み込みます（`frame_offset`や`num_frames`引数で部分読み込み可能） ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=,skip%20before%20start%20reading%20data))。PyTorchのTensorとして得られるため、そのままGPUに載せたりミニバッチ化したりといった操作がスムーズに行えます。デフォルト引数`channels_first=True`により、戻りTensorのshapeは`[チャンネル, 時間フレーム]`となります ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source))。

- **正規化**: torchaudioの`load`は**デフォルトで正規化を行います**が、その内容は「整数PCMを浮動小数に変換する」ことです ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source)) ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20,True))。`normalize=True`（デフォルト）の場合、整数型のWAVはtorch.float32テンソルに変換され、値が[-1.0,1.0]程度の範囲にスケーリングされます ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source))。これは**ボリューム正規化ではなく型変換**である点に注意が必要です ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Warning))。一方、引数`normalize=False`を指定すると、入力が整数PCMの場合は**元のビット深度の整数Tensor**が得られます ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20the%20input%20format%20is,tensors))。例えば16bit PCMならdtype=torch.int16、24bit PCMならint24がないためint32、32bit PCMならint32のTensorとなります ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20the%20input%20format%20is,tensors))。この場合、値は-32768～32767等の整数範囲を保持しています。浮動小数点WAVやFLAC/MP3などの場合、`normalize`フラグに関係なく**float32 Tensor**で読み込まれます ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=,mp3))。つまり、torchaudioは必要に応じて**柔軟にデータ型と正規化有無を選べる**設計です。

- **出力形式**: **PyTorchのTensor**で出力されます。デフォルトでは`float32`型で、shapeは(**チャンネル**, **フレーム数**)です ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source))。チャンネル数1の場合も2次元（1, N）のTensorになります（PyTorchでは1次元Tensorだとチャンネル軸情報を失うため）。`channels_first=False`を指定すれば(shape: **時間×チャンネル**)のTensorになります ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=WAV%20type))。NumPy配列が必要な場合は`.numpy()`メソッドで変換可能ですが、torchaudioはTensorで統一することでその後のニューラルネットワーク処理などへの連携をスムーズにしています。また、後述の通りSoXやSoundFileなど複数のバックエンドを選択可能ですが、出力の形式は統一されています。

- **対応フォーマット**: torchaudioは内部で複数のI/Oバックエンドを持ち、**デフォルトではSoX** (torchaudioソースコードに組み込まれたSoX-IO) を使用します。SoXはWAV (PCM/浮動小数点)はもちろん、MP3やFLACなど主要なフォーマットに対応しています。さらに**FFmpeg**をバックエンドに指定することもでき、これを用いればFFmpeg対応フォーマット全般を読み込めます ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=,%E2%80%93))。また、PySoundFileをバックエンドに選ぶことも可能です ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=,%E2%80%93))。標準のSoXバックエンドでも24bit PCMや32bit float PCMに対応しており、前述のように24bitはint32 Tensorにマップされます ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20the%20input%20format%20is,tensors))。チャンネル数やサンプリングレートもファイルから取得し、そのまま`sr`として返します。多チャンネル音声も問題なくTensor化できます。要約すると、**torchaudioは非常に幅広い音声フォーマットに対応**しており、特にWAVに関して不足はありません。

- **依存関係**: **PyTorch**本体と、その音声拡張ライブラリであるtorchaudioを必要とします。PyTorch自体が大規模なため、単に音声読み込みだけの目的で導入するのは重量級です。しかし既にPyTorchを使用しているプロジェクトでは追加負荷なく利用できます。torchaudio内部では、C++で実装されたSoXライブラリやFFmpegラッパーを利用しており、これらはtorchaudioインストール時に同梱されています。そのため、ユーザが別途SoXやFFmpegをインストールする必要は基本的にありません（ただしFFmpegバックエンド利用時は環境によってはシステムのffmpegを参照する場合もあります）。要するに、**機械学習用途向けにPyTorch + torchaudio環境が前提**となります。

- **他ライブラリとの統合**: torchaudioは**PyTorchとの統合**が最大の特徴です。読み込んだTensorはそのままGPU上のモデルに入力したり、PyTorchのDataset/DataLoaderで複数の音声をミニバッチ化したりできます。また、torchaudio自体にSTFTやフィルタ、データ増強（時間伸縮・ピッチシフト等）などの機能も揃っており、librosaに頼らずPyTorch内で完結した音響前処理が可能です。さらに、torchaudioはSoundFileをバックエンドにできるように設計されており、SoundFile経由で読み込んだNumPy配列をtorch.Tensorに変換する処理を内部的に行うこともできます。そのため、**他のライブラリとのデータ受け渡し**も比較的容易です（例：SoundFileで読みNumPy配列化 -> `torch.from_numpy()`でTensor化）。ただし基本的にはPyTorchユーザ以外にtorchaudioを使うメリットは少なく、独立した音声処理ライブラリというよりは**PyTorchユーザ向けの専用ツール**と言えます。

- **読み込み速度・パフォーマンス**: torchaudioは**高いパフォーマンス**を発揮します。C++実装のSoX or FFmpegを直接呼び出してデコードし、その結果を即座にTensor領域に格納するため、中間のPython処理が極力排されています。特に大量のファイルをGPU上で処理する深層学習ワークロードでは、データローディングがボトルネックになりがちですが、torchaudio + DataLoaderの組み合わせはマルチスレッドで効率よくバッチを組み立てることが可能です。ベンチマークによれば、SoundFile+NumPyで読み込んでからTensor化するよりもtorchaudioで直接Tensorとして読み込む方が若干高速であった例もあります ([Soundfile read is slower than torchaudio.load with soundfile backend · Issue #376 · bastibe/python-soundfile · GitHub](https://github.com/bastibe/python-soundfile/issues/376#:~:text=Interesting%20analysis,doing%20internally%20that%20is%20different))。これは内部でのメモリ転送や変換コストを削減しているためです。また、部分読み込み（frame_offset/num_frames）もサポートしており、大きなファイルから必要な区間だけ取得して訓練する、といった効率的なI/Oもできます。総じて、**大量データ処理に耐えうる設計と速度**を持っている点がtorchaudioの強みです。

**読み込みサンプルコード**:

```python
import torchaudio

# WAVファイルを読み込み（torch.Tensorで取得）
waveform, sr = torchaudio.load('input.wav')
print(sr)                # サンプリング周波数
print(waveform.shape, waveform.dtype)  
# 例: torch.Size([2, 441000]), torch.float32
# 必要に応じて NumPy 配列に変換:
np_wave = waveform.numpy()
```

## pydub

- **読み込み方法**: **pydub**は音声ファイルのカット・結合やエフェクト処理に便利な高レベルライブラリです。読み込みには`AudioSegment.from_wav('ファイルパス')`などのクラスメソッドを使用します。WAVに限らず、`from_file`で拡張子に応じたファイルを読み込めます。戻り値は**AudioSegmentオブジェクト**で、波形データやメタ情報を保持しています。直接NumPy配列ではありませんが、`audio_segment.get_array_of_samples()`メソッドで内部のPCMデータにアクセスできます ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=AudioSegment%28%E2%80%A6%29))。このメソッドはPythonの`array`型（標準配列モジュール）を返し、各サンプル値のシーケンスを取得できます ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=AudioSegment%28%E2%80%A6%29))。例えばステレオ音声なら、「左1サンプル, 右1サンプル, 左2サンプル, 右2サンプル,...」の順で並んだarray('h')（int16型配列）が得られます ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=Returns%20the%20raw%20audio%20data,sample_1_L%2C%20sample_1_R%2C%20sample_2_L%2C%20sample_2_R%2C%20%E2%80%A6))。それを`np.array()`でNumPy配列に変換すれば数値演算に使えます。

- **正規化**: **自動正規化は行われません**。AudioSegment内部ではソースのPCM値をそのまま保持しています。`get_array_of_samples()`が返す配列の値は、例えば16bit音源なら-32768～32767の整数値です。振幅を0～±1の実数に正規化するには、自身で`numpy.array(..., dtype=np.float32) / 32768.0`といった処理を行う必要があります ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=sound%20%3D%20sound,get_array_of_samples%28%29%20for%20s%20in%20channel_sounds))。pydubは主にオーディオ編集用途を想定しており、ラウドネス正規化などは別途メソッド（`apply_gain`等）で提供していますが、波形そのものの数値正規化は組み込まれていません。従って、分析のためにデータを取り出す際は**必要に応じて手動でスケーリング**します。

- **出力形式**: `get_array_of_samples()`の戻り値は**Pythonのarray型**（例えば`array('h')`）です ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=AudioSegment%28%E2%80%A6%29))。これをNumPyに変換すると**1次元のint型配列**になります。モノラル音声ならそのまま時系列配列ですが、ステレオなどの場合は左右チャネルが交互に並んだ1次元配列となるため、NumPy配列にした後で`.reshape(-1, 2)`のように2次元に整形する必要があります。この操作によりshape=(Nsamples, 2)の配列（列ごとに左/右チャンネル）が得られます。pydub自体はチャンネルを分離するメソッドも提供しており、例えば`audio_segment.split_to_mono()`で各チャンネルのAudioSegmentを取り出すことも可能です。その場合それぞれに対して`get_array_of_samples()`を呼べばチャンネル別のデータを取得できます。データ型は通常`int16`相当ですが、ファイルのbit深度によります。もっとも、後述の通り内部的に多くの形式は16bitに変換されるため、実質常に16bit整数として扱われます。

- **対応フォーマット**: pydubは**FFmpeg**もしくは**libav**を利用してほぼ全ての音声フォーマットを読み込みます。`AudioSegment.from_file`は拡張子や引数`format`でファイル種別を判断し、適切にデコードを試みます。WAVに関してはPCMはもちろん、非PCMや圧縮音声(WAVラップのMP3など)もffmpeg経由で展開できます。チャンネル数やサンプリングレートもファイルから引き継がれ、`audio.frame_rate`, `audio.channels`プロパティで確認できます。**ただし内部表現は特定のフォーマットに統一**されます。デフォルトではPCMの場合はソースと同じbit深度を維持しますが、FFmpegのデコード出力によっては16bitに変換されることがあります。特に**24bit WAVは直接扱えず16bitに落とされる**ことが知られています ([python - Process 24bit wavs using pydub - Stack Overflow](https://stackoverflow.com/questions/48847933/process-24bit-wavs-using-pydub#:~:text=It%20has%20been%20some%20months,using%20that%20data))。pydub自体は24bit対応していないため、別途24bit→16bit変換が入ります。そのため高音質データの精度保持には不向きです。一般的な16bitステレオ44.1kHz WAVやMP3程度であれば問題なく扱えます。

- **依存関係**: **ffmpegもしくはlibavのインストールが必要**です。pydubはそれらのコマンドラインツールを呼び出してデコードを行います。Pythonライブラリ単体では動作せず、環境にffmpegが無い場合はエラーになります。pydub自体は純Pythonで軽量ですが、背後にプロ音響用の強力なツールを必要とします。また、一部機能ではPython標準の`audioop`モジュールを使用しています。これはC実装のため高速ですが、主にモノラル/ステレオの範囲でしか動作確認されていません。つまり、5.1chなど多チャンネル音声の処理は限定的です。総じて、**pydub = Pythonからffmpegを操るラッパー**と考えると分かりやすいでしょう。

- **他ライブラリとの統合**: pydubは**オーディオデータの加工や変換に特化**しており、例えば波形を切り出してファイルに保存したり、MP3に一括変換したりといった用途に便利です。他の分析ライブラリと直接統合する想定はあまりなく、NumPy配列が必要な場合は上述の手順で取り出す必要があります。pydubを用いてデータ増強（重ね合わせやフェード処理）を行い、その結果をNumPyにして機械学習に入力、といったパイプラインを構築することも可能ですが、リアルタイム性や大規模データセット処理には向きません。基本的には**単発の音声編集操作**や**ファイル形式変換**などに優れたツールです。他ライブラリとの連携はファイルの入出力経由で行うのが一般的です。

- **読み込み速度・パフォーマンス**: ffmpegをバックエンドに使うためデコードそのものは高速ですが、**一度全体をAudioSegment（メモリ上）に展開する**仕様のため、大きなファイルではメモリ消費が大きくなります。読み込み速度は、ffmpegプロセスの起動時間が加算されるため、小さなファイルを大量に処理する場合にはオーバーヘッドが無視できなくなります。逆に長時間のファイルであっても一括で読み込むため、その後の処理はメモリ内で高速に行えます（代償として最初に時間を要する）。メタデータを取得するだけでもデコードが走るため、ライブラリによってはpydubでの情報取得は他より遅いと報告されています ([GitHub - faroit/python_audio_loading_benchmark: Benchmark popular audio i/o packages](https://github.com/faroit/python_audio_loading_benchmark#:~:text=Getting%20metadata%20information))。総合すると、**対話的・単発的な処理には便利だが、効率面では専門ライブラリに劣る**部分があります。しかしpydubの強みは性能よりも「簡潔に高度な編集ができる」点にあり、用途によっては十分実用的な速度です。

**読み込みサンプルコード**:

```python
from pydub import AudioSegment
import numpy as np

# WAVファイルをAudioSegmentとして読み込み
audio = AudioSegment.from_wav('input.wav')
samples = np.array(audio.get_array_of_samples())  # Python array -> NumPy配列
# ステレオの場合、2チャンネルに整形
if audio.channels > 1:
    samples = samples.reshape(-1, audio.channels)
print(audio.frame_rate, samples.shape, samples.dtype)
# 例: 44100, (441000, 2), int16
```

## ライブラリ比較まとめ

以上の各ライブラリの特徴を、**読み込み方法・APIの容易さ**, **正規化の挙動**, **出力データ形式**, **対応フォーマット（チャンネル数・サンプルレート・ビット深度）**, **依存関係**, **統合性**, **パフォーマンス**の観点でまとめた比較表を示します。

| ライブラリ              | 読み込み方法・APIの簡便さ                              | 正規化（振幅スケーリング）           | 出力形式                     | 対応フォーマット (チャンネル/サンプリングレート/ビット深度)                                        | 依存関係                          | 他ライブラリとの統合              | 読み込み速度・パフォーマンス               |
| ----------------------- | ---------------------------------------------------- | ------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------- | ---------------------------------------- |
| **SciPy.io.wavfile**    | 関数呼び出し一発で`sr`とデータ配列を取得。<br>低レベル機能のみで部分読み出し不可。 | **なし**（整数値そのまま）<br>※float PCMも値そのまま | NumPy配列（dtypeは入力依存：int16/int32/float32等） | PCMリニアWAVのみ対応。<br>モノ/ステレオ問わず任意チャンネル。<br>1～64bit整数PCM（24bitはint32に展開）対応 ([read — SciPy v1.15.2 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#:~:text=WAV%20files%20can%20specify%20arbitrary,bit%20and%20higher%20is%20signed))。<br>非PCMや圧縮WAVは不可。 | NumPy+SciPy（純Python実装）。<br>追加のC拡張不要。 | 出力がNumPyで汎用性高い。<br>他ライブラリへそのまま配列受け渡し可能。 | 純Python処理のため**中程度**。<br>大ファイルでは他より若干低速のことも。<br>メモリマップ可。 |
| **SoundFile**<br>(PySoundFile) | 関数で全読み込み、詳細制御も可能（開始位置・フレーム数指定等）。<br>ブロック読み出しも対応。 | **あり**（デフォルトfloat変換で-1～1スケーリング） ([python-soundfile — python-soundfile 0.13.1 documentation](https://python-soundfile.readthedocs.io/#:~:text=,optional%29%20%E2%80%93))<br>※dtype指定すれば整数そのまま | NumPy配列（既定float64）<br>※`dtype`でfloat32/int16等変更可 | WAVE/FLAC/OGGなど広範囲。<br>モノ～多チャンネル対応。<br>整数PCMは8～32bit、浮動小数点PCMもOK（libsndfile依存）。<br>24bitも完全サポート。 | **libsndfile**ライブラリ依存（C実装をCFFIで使用）。<br>pipで自動導入可。 | librosaが優先利用。<br>torchaudioでもバックエンド選択可。<br>NumPy出力で他ライブラリ連携容易。 | Cライブラリ利用で**高速**。<br>ディスクI/O以外のオーバーヘッド小。<br>部分読み込み可能で効率的。 |
| **librosa**            | `librosa.load`で簡単読み込み。<br>デフォルトで22050Hzに自動リサンプル ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Audio%20will%20be%20automatically%20resampled,sr%3D22050))・モノラル化 ([Multi-channel — librosa 0.10.2 documentation](http://librosa.org/doc/0.10.2/multichannel.html#:~:text=%28monaural%29%20signals%20are%20processed,in%20the%20signal%2C%20which%20is))。<br>パラメータ指定で原音忠実に可。 | **あり**（常にfloat32で-1～1正規化）<br>※内部でsoundfileまたは16bit変換処理 | NumPy配列（float32）<br>デフォルトmono時1次元、<br>ステレオ時shape=(2, N) ([Multi-channel — librosa 0.10.2 documentation](http://librosa.org/doc/0.10.2/multichannel.html#:~:text=%23%20Load%20as%20multi,load%28filename%2C%20mono%3DFalse)) | soundfileまたはaudioread経由で幅広く対応。<br>PCM WAVEはbit深度問わず可。<br>MP3等圧縮も読み込み可能（audioread使用）。<br>デフォルトでチャンネル平均化。 | **多数の依存**（NumPy/SciPy/others）。<br>soundfileまたはaudioread必須。<br>ライブラリサイズ大。 | 音響分析機能と一体化。<br>単体で完結するも、PyTorch等と組み合わせる場合は配列変換必要。 | リサンプル処理含め**まずまず高速**。<br>soundfile使用時はC実装で高速、<br>audioread使用時は多少オーバーヘッド。<br>大量処理ではやや重い。 |
| **wave** (標準)        | `wave.open`でファイル開封、`readframes`でbytes取得。<br>NumPy変換やチャンネル整形を自前実装する必要あり。 | **なし**（生のPCMバイト列） | bytes→ユーザが適宜int配列化。<br>例：int16のNumPy配列に手動変換。 | PCMエンコードWAVのみ ([wave — Read and write WAV files — Python 3.13.2 documentation](https://docs.python.org/3/library/wave.html#:~:text=The%20wave%20module%20provides%20a,encoded%20wave%20files%20are%20supported))。<br>Mono/Stereo他チャンネルOK（メタデータ取得可）。<br>8/16/24/32bit PCM対応（float非対応）。<br>圧縮音声は不可。 | **標準ライブラリのみ**。<br>追加依存なし。軽量。 | 汎用のbytes出力。<br>NumPy/他FWで使うには追加処理必要。<br>基本的に単体利用。 | 処理は単純で**低オーバーヘッド**。<br>高速だが、複数ファイル処理ではPython処理の負荷累積。<br>大量用途では非効率。 |
| **audioread**          | `audio_open`でファイル取得、forループでバッファ取得。<br>配列化はユーザ側で実装。手順多め。 | **なし**（16bit PCMに統一） ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=Buffers%20in%20the%20file%20can,to%20most%20of%20the%20backends))<br>※精度は16bit相当に丸め ([librosa.load — librosa 0.11.0 documentation](https://librosa.org/doc/main/generated/librosa.load.html#:~:text=Note)) | bytesチャンク（一括では返さない）。<br>最終的にint16のNumPy配列等に手動変換。 | **ほぼ全ての音声形式**に対応（FFmpeg/GStreamer依存） ([audioread · PyPI](https://pypi.org/project/audioread/#:~:text=,bindings))。<br>WAV含むあらゆるフォーマットを16bitPCMに変換出力。<br>チャンネル数・サンプルレートはファイル依存で取得可。 | **外部デコーダ依存**（ffmpeg等要インストール）。<br>Pythonパッケージ自体は軽量。 | librosa<=0.9で標準利用。<br>単体利用は稀。<br>他ライブラリの裏方的役割。 | デコード自体はFFmpeg等で**高速**。<br>ただPythonで小分けバッファ処理するため大量時は非効率。<br>16bit固定による情報損失あり。 |
| **torchaudio**         | `torchaudio.load`でTensor取得。<br>オプションで部分読み込みやchannel順変更可。<br>PyTorchコードと統一的に使える。 | **あり**（デフォルトで整数→float32正規化） ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=Load%20audio%20data%20from%20source))<br>※`normalize=False`指定で整数そのままTensor出力 ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20the%20input%20format%20is,tensors)) | **PyTorch Tensor** (既定float32) 。<br>shape=(channels, time)。<br>必要に応じnumpy変換可能。 | SoX/FFmpeg/SoundFileバックエンドによる広範囲フォーマット対応 ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=,%E2%80%93))。<br>WAV(PCM/Float)は24bit含め完全対応 ([torchaudio.load — Torchaudio 2.6.0 documentation](https://pytorch.org/audio/stable/generated/torchaudio.load.html#:~:text=When%20the%20input%20format%20is,tensors))。<br>MP3等もバックエンド次第で可。<br>多チャンネル・高サンプルレートOK。 | **PyTorch + torchaudio**が必要。<br>（裏でSoX/FFmpeg組み込み。ユーザが用意不要） | PyTorchと強力に統合。<br>テンソルをそのままNNに入力可能。<br>音響変換・フィルタ等もtorchaudio内で提供。 | C++実装で**極めて高速**。<br>大量データのバッチ読み込みに最適。<br>マルチスレッド読み込み可能。<br>部分読み込みも効率的。 |
| **pydub**              | `AudioSegment.from_file`で読み込み。<br>`AudioSegment`オブジェクト経由で操作。<br>`get_array_of_samples()`で配列取得。 | **なし**（整数PCM値を保持）。<br>正規化はユーザが必要なら実施 ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=sound%20%3D%20sound,get_array_of_samples%28%29%20for%20s%20in%20channel_sounds))。 | Python array経由でint配列取得。<br>モノラル:1次元、ステレオ:交互サンプル配列（整形必要） ([pydub/API.markdown at master · jiaaro/pydub · GitHub](https://github.com/jiaaro/pydub/blob/master/API.markdown#:~:text=Returns%20the%20raw%20audio%20data,sample_1_L%2C%20sample_1_R%2C%20sample_2_L%2C%20sample_2_R%2C%20%E2%80%A6))。 | FFmpeg対応フォーマットは全て可。<br>WAVはPCM全般対応だが、内部処理は16bitに集約（24bitは16bit化） ([python - Process 24bit wavs using pydub - Stack Overflow](https://stackoverflow.com/questions/48847933/process-24bit-wavs-using-pydub#:~:text=It%20has%20been%20some%20months,using%20that%20data))。<br>チャンネル数・サンプルレートも原則維持。 | **ffmpeg/libav必須**（外部ツール）。<br>pydub自体は純Python。 | 音声編集用に設計。<br>配列取得して他ライブラリで分析は可能だが手間。<br>ファイル入出力を介した連携が主。 | ffmpegデコードで**速度自体は高速**。<br>ただ全データ展開のため大規模処理には非効率。<br>小規模編集では十分な性能。 |

