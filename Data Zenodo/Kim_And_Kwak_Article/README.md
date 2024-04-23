models/research/audioset/vggish at master · tensorflow/models
https://github.com/tensorflow/models/tree/master/research/audioset/vggish

(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article (f531d50e ✗) # For Python 3 (which is recommended), you can create a virtual environment by:
python3 -m venv vggish_env

# Activate the virtual environment
source vggish_env/bin/activate

(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article (f531d50e ✗) python -m pip install --upgrade pip wheel

Requirement already satisfied: pip in ./vggish_env/lib/python3.8/site-packages (23.0.1)
Collecting pip
  Using cached pip-24.0-py3-none-any.whl (2.1 MB)
Collecting wheel
  Downloading wheel-0.43.0-py3-none-any.whl (65 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.8/65.8 kB 941.5 kB/s eta 0:00:00
Installing collected packages: wheel, pip
  Attempting uninstall: pip
    Found existing installation: pip 23.0.1
    Uninstalling pip-23.0.1:
      Successfully uninstalled pip-23.0.1
Successfully installed pip-24.0 wheel-0.43.0
(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article (f531d50e ✗) # Navigate to the directory where you will clone the TensorFlow models repository
cd path/to/your/project/directory

# Clone the TensorFlow models repository
git clone https://github.com/tensorflow/models.git

# Navigate to the VGGish directory
cd models/research/audioset/vggish

# Install the requirements
pip install -r requirements.txt

cd: no such file or directory: path/to/your/project/directory
Cloning into 'models'...
remote: Enumerating objects: 97112, done.
remote: Counting objects: 100% (392/392), done.
remote: Compressing objects: 100% (189/189), done.
remote: Total 97112 (delta 218), reused 346 (delta 202), pack-reused 96720
Receiving objects: 100% (97112/97112), 611.13 MiB | 910.00 KiB/s, done.
Resolving deltas: 100% (70636/70636), done.
Collecting numpy (from -r requirements.txt (line 1))
  Using cached numpy-1.24.4-cp38-cp38-macosx_11_0_arm64.whl.metadata (5.6 kB)
Collecting resampy (from -r requirements.txt (line 2))
  Using cached resampy-0.4.3-py3-none-any.whl.metadata (3.0 kB)
Collecting tensorflow (from -r requirements.txt (line 3))
  Downloading tensorflow-2.13.1-cp38-cp38-macosx_12_0_arm64.whl.metadata (2.6 kB)
Collecting tf_slim (from -r requirements.txt (line 4))
  Downloading tf_slim-1.1.0-py2.py3-none-any.whl.metadata (1.6 kB)
Collecting six (from -r requirements.txt (line 5))
  Using cached six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
Collecting soundfile (from -r requirements.txt (line 6))
  Using cached soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (14 kB)
Collecting numba>=0.53 (from resampy->-r requirements.txt (line 2))
  Downloading numba-0.58.1-cp38-cp38-macosx_11_0_arm64.whl.metadata (2.7 kB)
Collecting importlib-resources (from resampy->-r requirements.txt (line 2))
  Using cached importlib_resources-6.4.0-py3-none-any.whl.metadata (3.9 kB)
INFO: pip is looking at multiple versions of tensorflow to determine which version is compatible with other requirements. This could take a while.
Collecting tensorflow (from -r requirements.txt (line 3))
  Downloading tensorflow-2.13.0-cp38-cp38-macosx_12_0_arm64.whl.metadata (2.6 kB)
Collecting tensorflow-macos==2.13.0 (from tensorflow->-r requirements.txt (line 3))
  Using cached tensorflow_macos-2.13.0-cp38-cp38-macosx_12_0_arm64.whl.metadata (3.2 kB)
Collecting absl-py>=1.0.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
Collecting astunparse>=1.6.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=23.1.21 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
Collecting gast<=0.4.0,>=0.2.1 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
Collecting google-pasta>=0.1.1 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting h5py>=2.9.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Downloading h5py-3.11.0-cp38-cp38-macosx_11_0_arm64.whl.metadata (2.5 kB)
Collecting libclang>=13.0.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting numpy (from -r requirements.txt (line 1))
  Downloading numpy-1.24.3-cp38-cp38-macosx_11_0_arm64.whl.metadata (5.6 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Collecting packaging (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached packaging-24.0-py3-none-any.whl.metadata (3.2 kB)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Requirement already satisfied: setuptools in /Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3)) (56.0.0)
Collecting termcolor>=1.1.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
Collecting typing-extensions<4.6.0,>=3.6.6 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached typing_extensions-4.5.0-py3-none-any.whl.metadata (8.5 kB)
Collecting wrapt>=1.11.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached wrapt-1.16.0-cp38-cp38-macosx_11_0_arm64.whl.metadata (6.6 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Downloading grpcio-1.62.2-cp38-cp38-macosx_10_10_universal2.whl.metadata (4.0 kB)
Collecting tensorboard<2.14,>=2.13 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached tensorboard-2.13.0-py3-none-any.whl.metadata (1.8 kB)
Collecting tensorflow-estimator<2.14,>=2.13.0 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached tensorflow_estimator-2.13.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting keras<2.14,>=2.13.1 (from tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached keras-2.13.1-py3-none-any.whl.metadata (2.4 kB)
Collecting cffi>=1.0 (from soundfile->-r requirements.txt (line 6))
  Downloading cffi-1.16.0.tar.gz (512 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 512.9/512.9 kB 1.3 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting pycparser (from cffi>=1.0->soundfile->-r requirements.txt (line 6))
  Using cached pycparser-2.22-py3-none-any.whl.metadata (943 bytes)
Collecting llvmlite<0.42,>=0.41.0dev0 (from numba>=0.53->resampy->-r requirements.txt (line 2))
  Downloading llvmlite-0.41.1-cp38-cp38-macosx_11_0_arm64.whl.metadata (4.8 kB)
Collecting importlib-metadata (from numba>=0.53->resampy->-r requirements.txt (line 2))
  Using cached importlib_metadata-7.1.0-py3-none-any.whl.metadata (4.7 kB)
Collecting zipp>=3.1.0 (from importlib-resources->resampy->-r requirements.txt (line 2))
  Using cached zipp-3.18.1-py3-none-any.whl.metadata (3.5 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3)) (0.43.0)
Collecting google-auth<3,>=1.6.3 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached google_auth-2.29.0-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached google_auth_oauthlib-1.0.0-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting markdown>=2.6.8 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
Collecting requests<3,>=2.21.0 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached werkzeug-3.0.2-py3-none-any.whl.metadata (4.1 kB)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached pyasn1_modules-0.4.0-py3-none-any.whl.metadata (3.4 kB)
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached charset_normalizer-3.3.2-cp38-cp38-macosx_11_0_arm64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached MarkupSafe-2.1.5-cp38-cp38-macosx_10_9_universal2.whl.metadata (3.0 kB)
Collecting pyasn1<0.7.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached pyasn1-0.6.0-py2.py3-none-any.whl.metadata (8.3 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.14,>=2.13->tensorflow-macos==2.13.0->tensorflow->-r requirements.txt (line 3))
  Using cached oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
Using cached resampy-0.4.3-py3-none-any.whl (3.1 MB)
Downloading tensorflow-2.13.0-cp38-cp38-macosx_12_0_arm64.whl (1.9 kB)
Using cached tensorflow_macos-2.13.0-cp38-cp38-macosx_12_0_arm64.whl (189.3 MB)
Downloading numpy-1.24.3-cp38-cp38-macosx_11_0_arm64.whl (13.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.8/13.8 MB 2.0 MB/s eta 0:00:00
Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 352.1/352.1 kB 1.6 MB/s eta 0:00:00
Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Using cached soundfile-0.12.1-py2.py3-none-macosx_11_0_arm64.whl (1.1 MB)
Using cached absl_py-2.1.0-py3-none-any.whl (133 kB)
Downloading numba-0.58.1-cp38-cp38-macosx_11_0_arm64.whl (2.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 MB 1.5 MB/s eta 0:00:00
Using cached importlib_resources-6.4.0-py3-none-any.whl (38 kB)
Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Using cached flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
Using cached gast-0.4.0-py3-none-any.whl (9.8 kB)
Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
Downloading grpcio-1.62.2-cp38-cp38-macosx_10_10_universal2.whl (10.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.1/10.1 MB 1.8 MB/s eta 0:00:00
Downloading h5py-3.11.0-cp38-cp38-macosx_11_0_arm64.whl (2.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 1.4 MB/s eta 0:00:00
Using cached keras-2.13.1-py3-none-any.whl (1.7 MB)
Using cached libclang-18.1.1-py2.py3-none-macosx_11_0_arm64.whl (26.4 MB)
Downloading llvmlite-0.41.1-cp38-cp38-macosx_11_0_arm64.whl (28.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28.8/28.8 MB 1.4 MB/s eta 0:00:00
Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
Using cached protobuf-4.25.3-cp37-abi3-macosx_10_9_universal2.whl (394 kB)
Using cached tensorboard-2.13.0-py3-none-any.whl (5.6 MB)
Using cached tensorflow_estimator-2.13.0-py2.py3-none-any.whl (440 kB)
Using cached termcolor-2.4.0-py3-none-any.whl (7.7 kB)
Using cached typing_extensions-4.5.0-py3-none-any.whl (27 kB)
Using cached wrapt-1.16.0-cp38-cp38-macosx_11_0_arm64.whl (38 kB)
Using cached zipp-3.18.1-py3-none-any.whl (8.2 kB)
Using cached importlib_metadata-7.1.0-py3-none-any.whl (24 kB)
Using cached packaging-24.0-py3-none-any.whl (53 kB)
Using cached pycparser-2.22-py3-none-any.whl (117 kB)
Using cached google_auth-2.29.0-py2.py3-none-any.whl (189 kB)
Using cached google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
Using cached Markdown-3.6-py3-none-any.whl (105 kB)
Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Using cached tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Using cached cachetools-5.3.3-py3-none-any.whl (9.3 kB)
Using cached certifi-2024.2.2-py3-none-any.whl (163 kB)
Using cached charset_normalizer-3.3.2-cp38-cp38-macosx_11_0_arm64.whl (119 kB)
Downloading idna-3.7-py3-none-any.whl (66 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.8/66.8 kB 1.6 MB/s eta 0:00:00
Using cached MarkupSafe-2.1.5-cp38-cp38-macosx_10_9_universal2.whl (18 kB)
Using cached pyasn1_modules-0.4.0-py3-none-any.whl (181 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached rsa-4.9-py3-none-any.whl (34 kB)
Using cached urllib3-2.2.1-py3-none-any.whl (121 kB)
Using cached oauthlib-3.2.2-py3-none-any.whl (151 kB)
Using cached pyasn1-0.6.0-py2.py3-none-any.whl (85 kB)
Building wheels for collected packages: cffi
  Building wheel for cffi (pyproject.toml) ... done
  Created wheel for cffi: filename=cffi-1.16.0-cp38-cp38-macosx_11_0_arm64.whl size=170350 sha256=b2c7e70b4613234e7c91f4d10ff33844860bcfcbf54ef104c153aec64d3e0466
  Stored in directory: /Users/deangladish/Library/Caches/pip/wheels/f4/df/d7/20c740c0373c550cdca4fcf0eb9af36c769ad8553ea81c6a2f
Successfully built cffi
Installing collected packages: libclang, flatbuffers, zipp, wrapt, urllib3, typing-extensions, termcolor, tensorflow-estimator, tensorboard-data-server, six, pycparser, pyasn1, protobuf, packaging, oauthlib, numpy, MarkupSafe, llvmlite, keras, idna, grpcio, gast, charset-normalizer, certifi, cachetools, absl-py, werkzeug, tf_slim, rsa, requests, pyasn1-modules, opt-einsum, importlib-resources, importlib-metadata, h5py, google-pasta, cffi, astunparse, soundfile, requests-oauthlib, numba, markdown, google-auth, resampy, google-auth-oauthlib, tensorboard, tensorflow-macos, tensorflow
Successfully installed MarkupSafe-2.1.5 absl-py-2.1.0 astunparse-1.6.3 cachetools-5.3.3 certifi-2024.2.2 cffi-1.16.0 charset-normalizer-3.3.2 flatbuffers-24.3.25 gast-0.4.0 google-auth-2.29.0 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 grpcio-1.62.2 h5py-3.11.0 idna-3.7 importlib-metadata-7.1.0 importlib-resources-6.4.0 keras-2.13.1 libclang-18.1.1 llvmlite-0.41.1 markdown-3.6 numba-0.58.1 numpy-1.24.3 oauthlib-3.2.2 opt-einsum-3.3.0 packaging-24.0 protobuf-4.25.3 pyasn1-0.6.0 pyasn1-modules-0.4.0 pycparser-2.22 requests-2.31.0 requests-oauthlib-2.0.0 resampy-0.4.3 rsa-4.9 six-1.16.0 soundfile-0.12.1 tensorboard-2.13.0 tensorboard-data-server-0.7.2 tensorflow-2.13.0 tensorflow-estimator-2.13.0 tensorflow-macos-2.13.0 termcolor-2.4.0 tf_slim-1.1.0 typing-extensions-4.5.0 urllib3-2.2.1 werkzeug-3.0.2 wrapt-1.16.0 zipp-3.18.1
(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/models/research/audioset/vggish (master ✔) # Download the VGGish model checkpoint
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt

# Download the PCA parameters
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  277M  100  277M    0     0  2097k      0  0:02:15  0:02:15 --:--:-- 2290k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 73020  100 73020    0     0   180k      0 --:--:-- --:--:-- --:--:--  180k
(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/models/research/audioset/vggish (master ✗) python vggish_smoke_test.py


Testing your install of VGGish

Resampling via resampy works!
Log Mel Spectrogram example:  [[-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]
 [-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]
 [-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]
 ...
 [-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]
 [-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]
 [-4.48313252 -4.27083405 -4.17064267 ... -4.60069383 -4.60098887
  -4.60116305]]
/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer_v1.py:1697: UserWarning: `layer.apply` is deprecated and will be removed in a future version. Please use `layer.__call__` method instead.
  warnings.warn('`layer.apply` is deprecated and '
/Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages/tensorflow/python/keras/legacy_tf_layers/core.py:325: UserWarning: `tf.layers.flatten` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Flatten` instead.
  warnings.warn('`tf.layers.flatten` is deprecated and '
2024-04-22 21:11:16.602366: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:375] MLIR V1 optimization pass is not enabled
VGGish embedding:  [-2.72986382e-01 -1.80314213e-01  5.19921482e-02 -1.43571496e-01
 -1.04673788e-01 -4.96598274e-01 -1.75267994e-01  4.23148006e-01
 -8.22126269e-01 -2.16801286e-01 -1.17509425e-01 -6.70077145e-01
  1.43174559e-01 -1.44183934e-01  8.73502344e-03 -8.71974826e-02
 -1.84393570e-01  5.96655607e-01 -3.43809485e-01 -5.79105616e-02
 -1.65071458e-01  4.22910452e-02 -2.55293369e-01 -2.36356795e-01
  1.80295497e-01  3.02612185e-01  1.08356655e-01 -4.48397905e-01
  1.22757681e-01 -2.99955130e-01 -5.55934072e-01  5.05966544e-01
  2.05210537e-01  8.87591898e-01  9.03702378e-01 -2.10566476e-01
 -3.27462554e-02  1.38691589e-01 -2.27416530e-01  1.14804015e-01
  5.95409989e-01 -4.76971209e-01  2.28232548e-01  1.54626995e-01
  1.64934143e-01  7.19253063e-01  1.24101841e+00  5.61996222e-01
  2.73532122e-01  3.09789181e-02  2.10977659e-01 -6.09551787e-01
 -3.15282434e-01  1.76392794e-01 -8.96190405e-02 -4.26822513e-01
  3.12994003e-01 -1.56592414e-01  3.31673652e-01  1.29436389e-01
  1.66024059e-01  3.01902741e-02 -1.54465258e-01 -4.29332495e-01
 -2.68703699e-01 -1.58071116e-01  4.00485516e-01 -2.55945146e-01
 -2.66429223e-02  8.16181302e-03  2.98492849e-01  3.48756194e-01
 -1.07143715e-01  8.88779387e-02  1.26810521e-01 -3.34817320e-01
 -2.55427897e-01  5.07779598e-01  3.97584617e-01  1.78759545e-01
 -8.04520175e-02  4.84317839e-02 -2.01263011e-01 -2.97957540e-01
  3.66831362e-01  4.56224471e-01  5.37960529e-01 -2.00487375e-02
 -6.24543279e-02  4.15623158e-01 -1.88741565e-01 -5.36903262e-01
 -1.78362250e-01  3.81366968e-01  3.96644950e-01  3.21936488e-01
 -4.26684283e-02 -1.41018033e-01 -4.53833789e-01 -1.07017249e-01
 -2.21892640e-01  3.51183176e-01 -2.58386612e-01  3.31110179e-01
 -7.28939176e-01 -2.55487382e-01  3.56360823e-01 -3.16188395e-01
  3.12793732e-01  1.23501666e-01 -1.83649883e-02 -3.99395972e-01
 -5.13507247e-01 -2.74227172e-01 -2.68650711e-01  2.24091351e-01
  1.09624937e-01  1.30929962e-01 -1.25995010e-01 -1.92615181e-01
  1.83552504e-04  2.04150379e-01 -1.03096679e-01  2.93379426e-02
 -3.38305861e-01 -2.25750059e-01 -2.46723369e-01 -1.20763421e-01]
embedding mean/stddev 0.0006569798 0.34301957
Postprocessed VGGish embedding:  [160  53 124 132 154 120 119 105 155 173 129  69 149  93  59   0  52  97
 157 144 153 194 251 108  48 174 131 190 195  79  59  60 169  93 167 247
  28  75 255  56 134 169 234 137 232 100  19  80 162 255   0 255 101   0
 222 252  79 211  64  88 248   0   0 255 246  62  81 255   0 159  22 168
  70 255  99 135 204 192 255 150   0   0 255 255  67 235  55 255  69   0
   0  17 241  44 255 224   0 255  40   0 255   0 211 252  62   0  28 218
 112   0 255   0  81  67 153   0 255   0 129 229  53 255  55 101   0 255
   0 255]
postproc embedding mean/stddev 126.359375 89.33878063086252

Looks Good To Me!

(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/models/research/audioset/vggish (master ✗)
