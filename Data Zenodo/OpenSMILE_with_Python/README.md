(vggish_env) (tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) conda activate tensorflow
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) python audio_features.py
Traceback (most recent call last):
  File "audio_features.py", line 1, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) conda activate base
(base) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) python audio_features.py
Traceback (most recent call last):
  File "audio_features.py", line 1, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(base) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) conda activate tensorflow
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) python audio_features.py
Traceback (most recent call last):
  File "audio_features.py", line 1, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) pip install pandas
Collecting pandas
  Using cached pandas-2.0.3-cp38-cp38-macosx_11_0_arm64.whl.metadata (18 kB)
Requirement already satisfied: python-dateutil>=2.8.2 in /Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages (from pandas) (2.9.0.post0)
Collecting pytz>=2020.1 (from pandas)
  Using cached pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.1 (from pandas)
  Using cached tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
Requirement already satisfied: numpy>=1.20.3 in /Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages (from pandas) (1.24.3)
Requirement already satisfied: six>=1.5 in /Users/deangladish/CS-7643-O01/Group_Project/Data Zenodo/Kim_And_Kwak_Article/vggish_env/lib/python3.8/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
Using cached pandas-2.0.3-cp38-cp38-macosx_11_0_arm64.whl (10.7 MB)
Using cached pytz-2024.1-py2.py3-none-any.whl (505 kB)
Using cached tzdata-2024.1-py2.py3-none-any.whl (345 kB)
Installing collected packages: pytz, tzdata, pandas
Successfully installed pandas-2.0.3 pytz-2024.1 tzdata-2024.1
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗)





(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) python audio_features.py
python(24587,0x204c4a100) malloc: Heap corruption detected, free list is damaged at 0x600003138060
*** Incorrect guard value: 5070322176
python(24587,0x204c4a100) malloc: *** set a breakpoint in malloc_error_break to debug
[1]    24587 abort      python audio_features.py
(tensorflow) (vggish_env) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) deactivate
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗) python audio_features.py
                                                                                     F0semitoneFrom27.5Hz_sma3nz_amean  ...                  filename
file                                               start  end                                                           ...
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.503499999                          29.704147  ...  03-01-08-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.403395833                          27.940025  ...  03-01-08-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.503499999                          29.058771  ...  03-01-05-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.703687500                          27.157803  ...  03-01-06-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.237562500                          38.547504  ...  03-01-06-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.104104167                          41.331577  ...  03-01-05-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.870520833                          30.351290  ...  03-01-07-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.570229167                          23.949068  ...  03-01-04-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.737062500                          28.032854  ...  03-01-04-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.037375                             32.744171  ...  03-01-07-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.937270833                          31.276718  ...  03-01-03-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.470125                             28.093279  ...  03-01-03-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.703708333                          22.426426  ...  03-01-02-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.169833333                          24.268761  ...  03-01-01-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.503499999                          24.561495  ...  03-01-02-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.670333333                          34.288197  ...  03-01-03-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.536854167                          26.809052  ...  03-01-03-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.004000                             24.159096  ...  03-01-02-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.603583333                          23.718447  ...  03-01-02-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.303291667                          24.263884  ...  03-01-01-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.570229167                          31.813364  ...  03-01-08-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.269916667                          29.031378  ...  03-01-08-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.670333333                          28.219542  ...  03-01-06-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.937270833                          33.165081  ...  03-01-05-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.437770833                          41.773968  ...  03-01-05-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.970625                             37.544247  ...  03-01-06-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.503499999                          24.265032  ...  03-01-04-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.837166667                          26.670122  ...  03-01-07-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.104104167                          39.239143  ...  03-01-07-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.803791667                          25.748934  ...  03-01-04-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.470145833                          23.421627  ...  03-01-02-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.269916667                          24.186604  ...  03-01-01-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.004000                             23.797810  ...  03-01-02-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.470145833                          28.360966  ...  03-01-03-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.937250                             31.879890  ...  03-01-03-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.269916667                          25.457861  ...  03-01-08-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.269937500                          33.985863  ...  03-01-08-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.371020833                          33.348740  ...  03-01-07-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.737083333                          28.024416  ...  03-01-04-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.837166667                          25.805346  ...  03-01-04-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.903895833                          31.656343  ...  03-01-07-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.371041667                          37.747375  ...  03-01-05-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:05.005000                             35.420021  ...  03-01-06-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.436770833                          27.151884  ...  03-01-06-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.737062500                          29.093756  ...  03-01-05-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.536875                             28.157101  ...  03-01-08-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.436750                             29.426146  ...  03-01-08-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.703708333                          27.727692  ...  03-01-04-02-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.304291667                          37.311771  ...  03-01-07-02-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.903895833                          25.106548  ...  03-01-07-01-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.370041667                          24.396179  ...  03-01-04-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.903916667                          37.050087  ...  03-01-06-02-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.137458333                          36.234467  ...  03-01-05-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.870541667                          30.281694  ...  03-01-05-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.636979167                          30.818579  ...  03-01-06-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.336666667                          25.167889  ...  03-01-01-01-01-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.536854167                          24.898132  ...  03-01-02-01-01-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:04.204208333                          26.643515  ...  03-01-02-02-02-01-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.703708333                          26.414188  ...  03-01-03-01-02-02-01.wav
../Data Zenodo/Audio_Speech_Actors_01-24/Actor_... 0 days 0 days 00:00:03.636958333                          33.485363  ...  03-01-03-02-01-02-01.wav

[60 rows x 89 columns]
       F0semitoneFrom27.5Hz_sma3nz_amean  F0semitoneFrom27.5Hz_sma3nz_stddevNorm  ...  StddevUnvoicedSegmentLength  equivalentSoundLevel_dBp
count                          60.000000                               60.000000  ...                    60.000000                 60.000000
mean                           29.543520                                0.138691  ...                     0.444454                -40.424357
std                             4.966999                                0.061140  ...                     0.037243                  8.194434
min                            22.426426                                0.063495  ...                     0.320971                -52.330284
25%                            25.385367                                0.096009  ...                     0.429540                -46.924063
50%                            28.188321                                0.117856  ...                     0.449439                -41.648051
75%                            32.849397                                0.169115  ...                     0.467647                -35.666514
max                            41.773968                                0.318774  ...                     0.495101                -16.477798

[8 rows x 88 columns]
DataFrame loaded successfully. Available columns:
Index(['file', 'start', 'end', 'F0semitoneFrom27.5Hz_sma3nz_amean',
       'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
       'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
       'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
       'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
       'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
       'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
       'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
       'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
       'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope', 'loudness_sma3_amean',
       'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0',
       'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0',
       'loudness_sma3_pctlrange0-2', 'loudness_sma3_meanRisingSlope',
       'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope',
       'loudness_sma3_stddevFallingSlope', 'spectralFlux_sma3_amean',
       'spectralFlux_sma3_stddevNorm', 'mfcc1_sma3_amean',
       'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm',
       'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean',
       'mfcc4_sma3_stddevNorm', 'jitterLocal_sma3nz_amean',
       'jitterLocal_sma3nz_stddevNorm', 'shimmerLocaldB_sma3nz_amean',
       'shimmerLocaldB_sma3nz_stddevNorm', 'HNRdBACF_sma3nz_amean',
       'HNRdBACF_sma3nz_stddevNorm', 'logRelF0-H1-H2_sma3nz_amean',
       'logRelF0-H1-H2_sma3nz_stddevNorm', 'logRelF0-H1-A3_sma3nz_amean',
       'logRelF0-H1-A3_sma3nz_stddevNorm', 'F1frequency_sma3nz_amean',
       'F1frequency_sma3nz_stddevNorm', 'F1bandwidth_sma3nz_amean',
       'F1bandwidth_sma3nz_stddevNorm', 'F1amplitudeLogRelF0_sma3nz_amean',
       'F1amplitudeLogRelF0_sma3nz_stddevNorm', 'F2frequency_sma3nz_amean',
       'F2frequency_sma3nz_stddevNorm', 'F2bandwidth_sma3nz_amean',
       'F2bandwidth_sma3nz_stddevNorm', 'F2amplitudeLogRelF0_sma3nz_amean',
       'F2amplitudeLogRelF0_sma3nz_stddevNorm', 'F3frequency_sma3nz_amean',
       'F3frequency_sma3nz_stddevNorm', 'F3bandwidth_sma3nz_amean',
       'F3bandwidth_sma3nz_stddevNorm', 'F3amplitudeLogRelF0_sma3nz_amean',
       'F3amplitudeLogRelF0_sma3nz_stddevNorm', 'alphaRatioV_sma3nz_amean',
       'alphaRatioV_sma3nz_stddevNorm', 'hammarbergIndexV_sma3nz_amean',
       'hammarbergIndexV_sma3nz_stddevNorm', 'slopeV0-500_sma3nz_amean',
       'slopeV0-500_sma3nz_stddevNorm', 'slopeV500-1500_sma3nz_amean',
       'slopeV500-1500_sma3nz_stddevNorm', 'spectralFluxV_sma3nz_amean',
       'spectralFluxV_sma3nz_stddevNorm', 'mfcc1V_sma3nz_amean',
       'mfcc1V_sma3nz_stddevNorm', 'mfcc2V_sma3nz_amean',
       'mfcc2V_sma3nz_stddevNorm', 'mfcc3V_sma3nz_amean',
       'mfcc3V_sma3nz_stddevNorm', 'mfcc4V_sma3nz_amean',
       'mfcc4V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
       'hammarbergIndexUV_sma3nz_amean', 'slopeUV0-500_sma3nz_amean',
       'slopeUV500-1500_sma3nz_amean', 'spectralFluxUV_sma3nz_amean',
       'loudnessPeaksPerSec', 'VoicedSegmentsPerSec',
       'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec',
       'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength',
       'equivalentSoundLevel_dBp', 'filename'],
      dtype='object')
       F0semitoneFrom27.5Hz_sma3nz_amean  F0semitoneFrom27.5Hz_sma3nz_stddevNorm  ...  StddevUnvoicedSegmentLength  equivalentSoundLevel_dBp
count                          60.000000                               60.000000  ...                    60.000000                 60.000000
mean                           29.543520                                0.138691  ...                     0.444454                -40.424357
std                             4.966999                                0.061140  ...                     0.037243                  8.194434
min                            22.426426                                0.063495  ...                     0.320971                -52.330284
25%                            25.385367                                0.096009  ...                     0.429540                -46.924063
50%                            28.188321                                0.117856  ...                     0.449439                -41.648051
75%                            32.849397                                0.169115  ...                     0.467647                -35.666514
max                            41.773968                                0.318774  ...                     0.495101                -16.477798

[8 rows x 88 columns]
file                                       object
start                                      object
end                                        object
F0semitoneFrom27.5Hz_sma3nz_amean         float64
F0semitoneFrom27.5Hz_sma3nz_stddevNorm    float64
                                           ...
StddevVoicedSegmentLengthSec              float64
MeanUnvoicedSegmentLength                 float64
StddevUnvoicedSegmentLength               float64
equivalentSoundLevel_dBp                  float64
filename                                   object
Length: 92, dtype: object
Label column not found. Ensure you have the correct column name if you are conducting supervised learning.
Traceback (most recent call last):
  File "audio_features.py", line 269, in <module>
    correlation_matrix = df.corr()
  File "/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/pandas/core/frame.py", line 10054, in corr
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  File "/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/pandas/core/frame.py", line 1838, in to_numpy
    result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
  File "/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/pandas/core/internals/managers.py", line 1732, in as_array
    arr = self._interleave(dtype=dtype, na_value=na_value)
  File "/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/pandas/core/internals/managers.py", line 1794, in _interleave
    result[rl.indexer] = arr
ValueError: could not convert string to float: '../Data Zenodo/Audio_Speech_Actors_01-24/Actor_01/03-01-08-02-02-01-01.wav'
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo/OpenSMILE_with_Python (f531d50e ✗)
