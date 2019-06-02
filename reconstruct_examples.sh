cls="02691156 02828884 02933112 02958343 03001627 03211117 03636649 03691459 04090263 04256520 04379243 04401088 04530566"
index="0 1 2"

cls_arr=$cls
id_arr=$index

#echo $cls_arr
#echo $id_arr

for c in $cls_arr
do
    for id in $id_arr
    do
        echo $c $id
        python mesh_reconstruction/reconstruct.py -d ./data/models -eid singleclass_${c} -i ./data/test_img/${c}_${id}_in.png -oi ./data/test_img/${c}_${id}_out.png -oo ./data/test_img/${c}_${id}_out.obj
    done
done
