cuda=1


# for seed in 1 2 3; do
# for epoch in 500 1000 5000 10000 20000; do
# for lr in 0.001 0.003 0.005 0.008 0.01; do
# for wd in 5e-4 1e-5 1e-6; do
# # for loss in 'cross_entropy' 'nll_loss' 'mse_loss'
# for layers in 1 2 3 4; do
# for hidden in 8 16 32 64; do
# 	python src.py --epoch ${epoch} --layers ${layers} --hidden ${hidden} --lr ${lr} --wd ${wd} --seed ${seed}
# done
# done
# done
# done
# done
# done
# # done


# for seed in 1 2 3; do
# for epoch in 500 1000 10000 20000; do
# for lr in 0.001 0.005 0.01; do
for epoch in 20000; do
for lr in 0.005 0.01; do
for wd in 5e-4 1e-5 1e-6; do
# for loss in 'cross_entropy' 'nll_loss' 'mse_loss'
# for layers in 1 2 3 4; do
# for hidden in 8 16 32 64; do
	python src.py --epoch ${epoch} --lr ${lr} --wd ${wd}
done
done
done
# done
# done
# done
# done