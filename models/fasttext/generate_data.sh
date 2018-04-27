#echo 'preprocessing training data...'
#awk -F '   ' '{print $2}' /Users/Eric/Desktop/sohu_seg/raw_data/News_info_train_seg.txt | awk '{ s_count = 0; line=""; for(i=NF-1;i>0;i--) { if( $i == "," || $i == "，" || $i == "。" || $i == "！" || $i == "？" || $i == "；" || $i == "?" || $i == "!") s_count = s_count + 1 ; if(s_count < 5) line = $i" "line; else break; }; print line }' > news_content.train.raw

#echo 'removing stopwords...'
#awk -F " " 'NR==FNR {s[$1] = 1} NR != FNR {b="";for(i=1;i<=NF;i++) if(s[$i] !=1 ) b=b""$i" "; print b}' /Users/Eric/Desktop/sohu_seg/stopwords.txt  news_content.train.raw > news_content.train

if [ ! -d "data" ]; then
  mkdir data
fi

echo 'generate data...'
awk -F '\t' '{print $2}' ../../data/News_info_train_seg.txt > data/temp1

echo 'adding label...'
awk -F "\t" 'NR==FNR {a[NR] = $2} NR!=FNR {print "__label__"a[FNR]" "$0}' ../../data/News_pic_label_train.txt data/temp1 > data/temp2

#echo 'shuff data...'
#gshuf data/temp2 > data/temp3

echo 'splitting data...'
cat data/temp2 | head -8480 > data/X.validate
cat data/temp2 | tail -40000 > data/X.train

#echo 'preprocessing testing data...'
#awk -F '   ' '{print $2}' /Users/Eric/Desktop/sohu_seg/raw_data/News_info_validate_seg.txt | awk '{ s_count = 0; line=""; for(i=NF-1;i>0;i--) { if( $i == "," || $i == "，" || $i == "。" || $i == "！" || $i == "？" || $i == "；" || $i == "?    " || $i == "!") s_count = s_count + 1 ; if(s_count < 5) line = $i" "line; else break; }; print line }' > news.test.raw

#echo 'removing stopwords...'
#awk -F " " 'NR==FNR {s[$1] = 1} NR != FNR {b="";for(i=1;i<=NF;i++) if(s[$i] !=1 ) b=b""$i" "; print b}' /Users/Eric/Desktop/sohu_seg/stopwords.txt  news.test.raw > news.test

awk -F '\t' '{print $2}' ../../data/News_info_validate_seg.txt > data/X.test

rm data/temp*
echo 'Done!'

