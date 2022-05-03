TARGET='/mnt/d/Lab/projects/swerc-notebook'

cd $TARGET

find . -type f -name '*.cpp' | while read CPPFILE
do
    TITLE=$(basename $CPPFILE .cpp)
    echo $CPPFILE | xargs enscript --color=1 -C -Ecpp -B -t $TITLE -o - | ps2pdf - $TITLE.pdf
done
