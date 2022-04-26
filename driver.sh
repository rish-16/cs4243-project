python v2_sweep.py --channels 64 128 192 256 512 --dropout 0.3 --dim 256 --lr 0.025 >> v2_sweep_1.txt
python v2_sweep.py --channels 64 128 256 512 --dropout 0.5 --dim 512 --lr 0.02 >> v2_sweep_2.txt
python v2_sweep.py --channels 32 64 64 512 --dropout 0.4 --dim 128 --lr 0.05 >> v2_sweep_3.txt
python v2_sweep.py --channels 16 32 128 512 --dropout 0.3 --dim 256 --lr 0.02 >> v2_sweep_4.txt
python v2_sweep.py --channels 32 64 192 512 --dropout 0.5 --dim 512 --lr 0.04 >> v2_sweep_5.txt
python v2_sweep.py --channels  32 128 192 512 --dropout 0.5 --dim 128 --lr 0.002 >> v2_sweep_6.txt