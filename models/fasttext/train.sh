#word_n_grams=$2
#epoch=$1
datename=$(date +%Y%m%d-%H%M%S)
fasttext='/home/fanyy/spam_detection/fastText/fasttext'

echo 'training model...'
$fasttext supervised -input data/X.train -output model/model_news 

#-epoch $1 -wordNgrams $2
mkdir model

echo 'validating...'
$fasttext test model/model_news.bin data/X.validate | awk '{ b=b"_"$2} END{print b}'

echo 'making predictions...'
$fasttext predict model/model_news.bin data/X.test | sed 's/__label__//' > news.predict

#file_path="result_"$datename"_epoch_"$epoch"_ngrams_"$word_n_grams""$P".txt"
file_path="result_"$datename".txt"
echo 'generating final results...'
awk -F '\t' 'NR==FNR{a[FNR] = $1; b=FNR;} NR!=FNR{print $1 "\t"a[NR-b]"\t""NULL""\t""NULL"}' news.predict ../../data/News_info_validate_seg.txt > $file_path

rm news.predict

echo 'Done!'

