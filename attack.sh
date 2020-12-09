# MetaAttack source (Meta-Train):market1501, meta-test:personx, target:msmt17, remember to change the path of re-ID models
python -W ignore mainMAML.py -s market1501 -m personx -t msmt17 --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/msmt/msmt17.pth --noise_path ./noise_m2d
# MetaAttack source (Meta-Train):market1501, meta-test:personx, target:dukemtmc, remember to change the path of re-ID models
python -W ignore mainMAML.py -s market1501 -m personx -t dukemtmc --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/duke/duke.pth --noise_path ./noise_m2d

# Multi-source: market1501(Meta-Train), meta-test:personx, target:dukemtmc, remember to change the path of re-ID models
python -W ignore mainAdd.py -s market1501 -m personx -t dukemtmc --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/duke/duke.pth --noise_path ./noise_m2d
# Multi-source: market1501(Meta-Train), meta-test:personx, target:msmt, remember to change the path of re-ID models
python -W ignore mainAdd.py -s market1501 -m personx -t msmt17 --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/msmt/msmt17.pth --noise_path ./noise_m2d

# Attack with pair-wise and label-wise corruption (i.e., UAP-Retrieval, w/o list loss)
# source (Meta-Train): market1501, meta-test:personx, target:dukemtmc, remember to change the path of re-ID models
python -W ignore main.py -s market1501 -t dukemtmc --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/duke/duke.pth --noise_path ./noise_m2d
# Attack with pair-wise and label-wise corruption (i.e., UAP-Retrieval, w/o list loss)
# source (Meta-Train): market1501, meta-test:personx, target:msmt17, remember to change the path of re-ID models
python -W ignore main.py -s market1501 -t msmt17 --data ./data --batch_size 10 --resume logs/market/market.pth --resumeTgt logs/msmt/msmt17.pth --noise_path ./noise_m2d